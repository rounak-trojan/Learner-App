"""
Face detection + embedding engine.

Detection : MTCNN (facenet-pytorch)
Embedding : InceptionResnetV1 pretrained on VGGFace2 (512-d vector, facenet-pytorch)
Matching  : cosine similarity, per-center in-memory index (numpy)

Embeddings are the only thing persisted for a learner - never raw pixels.
This is intentional: embeddings are compact, fast to compare, and robust to
lighting/angle/expression changes in a way raw pixel comparison is not.
"""
import threading
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

import config

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_lock = threading.Lock()  # torch modules used from a single worker at a time is simplest for a demo server

# min_face_size lowered from the factory default (20) is what lets MTCNN keep candidates
# that are small in the frame because the person is standing farther from the camera.
# keep_all=True is what makes multi-face (group) detection work at all - it returns every
# face MTCNN finds above the probability threshold, not just the most prominent one.
_mtcnn = MTCNN(
    keep_all=True,
    device=_device,
    post_process=False,
    min_face_size=config.MTCNN_MIN_FACE_SIZE,
)
_resnet = InceptionResnetV1(pretrained="vggface2").eval().to(_device)


def _blur_score(gray_img):
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()


def enhance_frame(frame_bgr):
    """
    Preprocessing pass applied before detection, aimed at the two failure modes
    reported in the field: (1) faces far from the camera arrive as small, low-contrast
    regions, and (2) auto-exposure/auto-focus lag makes frames mildly blurry or dim.

    Pipeline:
      1. Upscale if the frame is smaller than ENHANCE_UPSCALE_MAX_WIDTH, so a face that's
         only ~40px wide at capture resolution has more pixels for MTCNN and the embedder
         to work with (this is what mainly helps "far away" capture, short of a better lens).
      2. CLAHE (adaptive local contrast) on the luma channel - recovers detail in dim/backlit
         group shots without blowing out already-bright areas the way global histogram
         equalization would.
      3. A light unsharp-mask pass to claw back edge definition lost to motion/focus blur.
    Runs on every frame before MTCNN sees it, so it benefits detection, blur scoring, and
    the embedding alike.
    """
    h, w = frame_bgr.shape[:2]
    if w < config.ENHANCE_UPSCALE_MAX_WIDTH:
        scale = min(2.0, config.ENHANCE_UPSCALE_MAX_WIDTH / w)
        if scale > 1.05:
            frame_bgr = cv2.resize(
                frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
            )

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    frame_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=1.2)
    frame_bgr = cv2.addWeighted(frame_bgr, 1.5, blurred, -0.5, 0)

    return frame_bgr


def detect_faces(frame_bgr, enhance=True, min_prob=None, min_face_px=None):
    """Detect faces in a BGR frame. Returns list of dicts: embedding, box, blur, prob.
    enhance=True runs the far-field/low-light enhancement pass first (see enhance_frame).
    keep_all on the underlying MTCNN means every face in a group shot comes back here,
    not just one - the caller (api_scan) loops over all of them so a class of 10-12
    walking past the camera together all get matched in the same request."""
    if enhance:
        frame_bgr = enhance_frame(frame_bgr)

    min_prob = config.MIN_DETECT_PROB if min_prob is None else min_prob
    min_face_px = 0 if min_face_px is None else min_face_px

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with _lock:
        boxes, probs = _mtcnn.detect(rgb)
    results = []
    if boxes is None:
        return results

    h, w = frame_bgr.shape[:2]
    for box, prob in zip(boxes, probs):
        if prob is None or prob < min_prob:
            continue
        x1, y1, x2, y2 = box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) < min_face_px or (y2 - y1) < min_face_px:
            continue
        face_crop = frame_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur = _blur_score(gray)

        face_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_CUBIC)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
        tensor = (tensor - 127.5) / 128.0
        tensor = tensor.unsqueeze(0).to(_device)

        with torch.no_grad(), _lock:
            emb = _resnet(tensor).cpu().numpy()[0]

        results.append(
            {
                "embedding": emb.astype(np.float32),
                "box": (x1, y1, x2, y2),
                "blur": float(blur),
                "prob": float(prob),
            }
        )
    return results


def detect_faces_for_scan(frame_bgr):
    """Attendance-scan variant: looser probability/size/blur gates than enrollment, because
    a group of people 2m+ back produces smaller, softer face crops than a single learner
    filling the frame during enrollment. Enhancement is always applied here."""
    return detect_faces(
        frame_bgr,
        enhance=True,
        min_prob=config.SCAN_MIN_DETECT_PROB,
        min_face_px=config.SCAN_MIN_FACE_PX,
    )


def _farthest_point_sampling(vectors, k):
    """Pick k diverse vectors out of the set (covers different pose/lighting better than
    taking the first k frames, which tend to look similar)."""
    n = len(vectors)
    if n <= k:
        return list(range(n))
    selected = [0]
    dists = np.linalg.norm(vectors - vectors[0], axis=1)
    for _ in range(k - 1):
        next_idx = int(np.argmax(dists))
        selected.append(next_idx)
        new_dists = np.linalg.norm(vectors - vectors[next_idx], axis=1)
        dists = np.minimum(dists, new_dists)
    return selected


def process_enrollment_video(video_path, progress_cb=None):
    """
    Walk through an enrollment video, pull good-quality single-face frames,
    embed them, and return a diverse subset of embeddings to persist.

    Assumes one learner per video (takes the largest face per frame, discards
    the rest - guards lightly against a bystander wandering into frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(fps * config.ENROLL_SAMPLE_SECONDS))

    candidate_embeddings = []
    idx = 0
    accepted = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            dets = detect_faces(frame)
            if dets:
                dets.sort(key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]), reverse=True)
                best = dets[0]
                if best["blur"] >= config.BLUR_THRESHOLD:
                    candidate_embeddings.append(best["embedding"])
                    accepted += 1
                    if progress_cb:
                        progress_cb(accepted)
        idx += 1
    cap.release()

    if not candidate_embeddings:
        return []

    arr = np.vstack(candidate_embeddings)
    keep_idx = _farthest_point_sampling(arr, config.MAX_EMBEDDINGS_PER_LEARNER)
    return [arr[i] for i in keep_idx]


def process_enrollment_frames(frames_bgr_list):
    """
    Same idea as process_enrollment_video, but the input is already a list of BGR
    frames captured live in the browser (the guided front/left/right registration
    scanner) instead of a video file on disk.

    Assumes one learner per capture session (takes the largest face per frame,
    discards the rest - guards lightly against a bystander wandering into frame).
    """
    candidate_embeddings = []
    for frame in frames_bgr_list:
        if frame is None:
            continue
        dets = detect_faces(frame)
        if not dets:
            continue
        dets.sort(key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]), reverse=True)
        best = dets[0]
        if best["blur"] >= config.BLUR_THRESHOLD:
            candidate_embeddings.append(best["embedding"])

    if not candidate_embeddings:
        return []

    arr = np.vstack(candidate_embeddings)
    keep_idx = _farthest_point_sampling(arr, config.MAX_EMBEDDINGS_PER_LEARNER)
    return [arr[i] for i in keep_idx]


class CenterIndex:
    """In-memory per-center embedding index for fast scan-time search.
    Keeps the scanner from ever comparing against learners outside the
    logged-in center's own repository."""

    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def build(self, center_uid, rows):
        """rows: list of (luid, vector_bytes)"""
        if not rows:
            with self._lock:
                self._cache[center_uid] = (np.array([]), np.zeros((0, 512), dtype=np.float32))
            return
        luids = []
        vecs = []
        for luid, blob in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            luids.append(luid)
            vecs.append(v)
        with self._lock:
            self._cache[center_uid] = (np.array(luids), np.vstack(vecs).astype(np.float32))

    def ensure_loaded(self, center_uid, loader_fn):
        if center_uid not in self._cache:
            self.build(center_uid, loader_fn(center_uid))

    def invalidate(self, center_uid):
        with self._lock:
            self._cache.pop(center_uid, None)

    def search(self, center_uid, query_embedding, threshold=None):
        threshold = config.MATCH_THRESHOLD if threshold is None else threshold
        if center_uid not in self._cache:
            return None
        luids, mat = self._cache[center_uid]
        if len(luids) == 0:
            return None
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        sims = mat @ q
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= threshold:
            return luids[best_idx], best_score
        return None, best_score


center_index = CenterIndex()
