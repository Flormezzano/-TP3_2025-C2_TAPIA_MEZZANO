#!/usr/bin/env python3
"""
TP3 - PDI (Problema 1 - Cinco dados) - Punto a

Consigna:
- Detectar automáticamente los frames donde los dados están detenidos
- Mostrar por terminal el valor obtenido en cada dado
- El script debe informar y mostrar resultados de cada etapa

Estrategia (simple + robusta a escala):
1) Detección de reposo: energía de movimiento por diferencia entre frames (absdiff) y umbral.
   - Trabajamos en una versión reescalada del video para estabilidad y velocidad.
   - Umbrales y tamaños de kernels se definen en función del tamaño de imagen (no áreas fijas).
2) Segmentación de dados: máscara por color (HSV) + componentes conectados.
   - El color del dado se estima automáticamente por histograma de hue en pixeles saturados.
3) Conteo de puntos (pips) dentro de cada dado:
   - Probamos dos hipótesis: pips claros (blancos) vs pips oscuros (negros).
   - Elegimos la que produce un conteo plausible (1..6) y con mejor “circularidad”.

Requisitos: opencv-python, numpy, matplotlib (opcional para plot)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import cv2

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


@dataclass
class Config:
    # --- reposo ---
    motion_resize_w: int = 540
    motion_diff_thr: int = 10                # umbral sobre absdiff (en 0..255)
    static_thr_percentile: int = 10          # percentil para definir umbral de “baja movilidad”
    min_static_sec: float = 0.30             # duración mínima del tramo “en reposo”
    # --- dados ---
    dice_resize_w: int = 720
    expected_dice: int = 5
    hue_delta: int = 10                      # ventana +/- alrededor del hue detectado
    sat_min: int = 80
    val_min: int = 40
    # --- pips ---
    pip_area_min_ratio: float = 0.0008       # respecto del ROI del dado
    pip_area_max_ratio: float = 0.03
    circularity_min: float = 0.35
    # --- debug ---
    debug: bool = True
    out_dir: str = "tp3_debug"


def resize_keep_aspect(img, target_w: int):
    h, w = img.shape[:2]
    if w == target_w:
        return img, 1.0
    scale = target_w / float(w)
    out = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    return out, scale


def circular_hue_dist(h, center):
    """Distancia circular en el espacio de hue OpenCV (0..179)."""
    d = np.abs(h.astype(np.int16) - int(center))
    return np.minimum(d, 180 - d)


def estimate_dice_hue(hsv, sat_min=80, val_min=40):
    """
    Estima el hue “no dominante de fondo”:
    - Toma pixeles saturados (probables colores)
    - Histograma por hue
    - Selecciona el 2do pico (el 1ro suele ser el fondo si ocupa mucho)
    """
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    mask = (S >= sat_min) & (V >= val_min)
    hvals = H[mask]
    if hvals.size < 500:
        # fallback: rojo típico (0)
        return 0

    hist = np.bincount(hvals, minlength=180).astype(np.float32)

    # Suavizado simple para estabilizar picos
    hist = np.convolve(hist, np.ones(5, dtype=np.float32) / 5.0, mode="same")

    peaks = np.argsort(hist)[::-1]
    peak1 = int(peaks[0])
    # Elegimos el primer pico que no sea "demasiado cercano" al dominante
    for p in peaks[1:10]:
        p = int(p)
        if min(abs(p - peak1), 180 - abs(p - peak1)) >= 15:
            return p
    return int(peaks[1])


def dice_mask_from_hue(hsv, hue_center, hue_delta, sat_min=80, val_min=40):
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    m = (circular_hue_dist(H, hue_center) <= hue_delta) & (S >= sat_min) & (V >= val_min)
    return (m.astype(np.uint8) * 255)


def components_from_mask(mask, expected=5):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = mask.shape[:2]
    frame_area = H * W

    cands = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        box_area = w * h
        ar = w / (h + 1e-6)

        # Filtros relativos al tamaño de imagen (no hardcodeamos áreas absolutas)
        if area < 0.0003 * frame_area or area > 0.05 * frame_area:
            continue
        if box_area < 0.0005 * frame_area or box_area > 0.08 * frame_area:
            continue
        if not (0.6 <= ar <= 1.6):
            continue

        cands.append((area, (x, y, w, h)))

    # nos quedamos con los más grandes
    cands = sorted(cands, key=lambda t: t[0], reverse=True)[: expected * 2]
    boxes = [b for _, b in sorted(cands, key=lambda t: (t[1][0], t[1][1]))]
    return boxes[:expected]


def detect_dice_boxes(frame_bgr, cfg: Config):
    img, _ = resize_keep_aspect(frame_bgr, cfg.dice_resize_w)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hue = estimate_dice_hue(hsv, sat_min=cfg.sat_min, val_min=cfg.val_min)
    mask = dice_mask_from_hue(hsv, hue, cfg.hue_delta, sat_min=cfg.sat_min, val_min=cfg.val_min)

    # Morfología: cerrar (unir) + abrir (limpiar)
    ksz = max(5, int(min(img.shape[:2]) * 0.01)) | 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    boxes = components_from_mask(mask, expected=cfg.expected_dice)
    return img, boxes, mask, hue


def circularity(area, perimeter):
    if perimeter <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def pip_candidates_from_mask(pip_mask, roi_area, cfg: Config):
    # Componentes conectados y filtrado por área + circularidad
    num, labels, stats, _ = cv2.connectedComponentsWithStats(pip_mask, connectivity=8)
    pips = 0
    circ_sum = 0.0

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < cfg.pip_area_min_ratio * roi_area or area > cfg.pip_area_max_ratio * roi_area:
            continue

        comp = ((labels == i).astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        per = cv2.arcLength(cnt, True)
        circ = circularity(area, per)
        if circ < cfg.circularity_min:
            continue

        pips += 1
        circ_sum += circ

    return pips, circ_sum


def count_pips_in_die(img_bgr_resized, box, cfg: Config):
    x, y, w, h = box
    pad = int(0.15 * max(w, h))
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, img_bgr_resized.shape[1])
    y1 = min(y + h + pad, img_bgr_resized.shape[0])

    roi = img_bgr_resized[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    # Máscaras candidatas:
    # 1) pips claros: baja saturación + alto valor
    s_thr = np.percentile(S, 35)
    v_thr = np.percentile(V, 70)
    pip_white = ((S < s_thr) & (V > v_thr)).astype(np.uint8) * 255

    # 2) pips oscuros: bajo valor (y saturación no muy alta)
    v_thr2 = np.percentile(V, 30)
    s_thr2 = np.percentile(S, 70)
    pip_black = ((V < v_thr2) & (S < s_thr2)).astype(np.uint8) * 255

    # Morfología en escala relativa al ROI
    ksz = max(3, int(min(roi.shape[:2]) * 0.05)) | 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))

    def clean(m):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
        return m

    pip_white = clean(pip_white)
    pip_black = clean(pip_black)

    roi_area = roi.shape[0] * roi.shape[1]
    n_w, score_w = pip_candidates_from_mask(pip_white, roi_area, cfg)
    n_b, score_b = pip_candidates_from_mask(pip_black, roi_area, cfg)

    def rank(n, score):
        # preferimos valores plausibles y más “círculos”
        plausible = 1 <= n <= 6
        return (1 if plausible else 0, score, -abs(n - 4))

    if rank(n_b, score_b) > rank(n_w, score_w):
        return n_b, pip_black
    return n_w, pip_white


def compute_motion_scores(video_path, cfg: Config):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scores = []

    prev_gray = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fr, _ = resize_keep_aspect(frame, cfg.motion_resize_w)
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is None:
            scores.append(1.0)
        else:
            diff = cv2.absdiff(gray, prev_gray)
            changed = (diff > cfg.motion_diff_thr).astype(np.uint8)
            score = float(changed.sum()) / float(changed.size)
            scores.append(score)

        prev_gray = gray

    cap.release()
    return np.array(scores, dtype=np.float32), float(fps)


def find_static_segments(scores, fps, cfg: Config):
    # Umbral “baja movilidad” por percentil (robusto entre videos)
    tail = scores[min(5, len(scores)):]
    thr = float(np.percentile(tail, cfg.static_thr_percentile))
    # Evitar el caso thr=0 ultra estricto si hay cuantización:
    if thr == 0.0:
        thr = float(np.percentile(tail, min(cfg.static_thr_percentile + 10, 30)))

    min_len = max(3, int(cfg.min_static_sec * fps))

    segs = []
    i = 0
    n = len(scores)
    while i < n:
        if scores[i] <= thr:
            j = i
            while j < n and scores[j] <= thr:
                j += 1
            if (j - i) >= min_len:
                segs.append((i, j))
            i = j
        else:
            i += 1

    return segs, thr


def pick_best_frame(video_path, seg, cfg: Config):
    start, end = seg
    best = None

    for idx in range(start, end):
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue

        img, boxes, mask, hue = detect_dice_boxes(frame, cfg)
        vals = []
        valid = 0

        for b in boxes:
            n, _ = count_pips_in_die(img, b, cfg)
            vals.append(n)
            if 1 <= n <= 6:
                valid += 1

        # scoring: primero que tenga 5 dados, luego que los valores sean válidos
        score = (len(boxes), valid)
        if best is None or score > best[0]:
            best = (score, idx, frame, img, boxes, vals, mask, hue)

        # si encontramos el caso perfecto, salimos
        if best and best[0] == (cfg.expected_dice, cfg.expected_dice):
            break

    if best is None:
        return (start + end) // 2, [], None

    _, idx, frame, img, boxes, vals, mask, hue = best
    debug_pack = dict(frame=frame, img=img, boxes=boxes, vals=vals, mask=mask, hue=hue)
    return idx, vals, debug_pack


def draw_debug(img, boxes, vals):
    out = img.copy()
    for (x, y, w, h), v in zip(boxes, vals):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, str(v), (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(v), (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def maybe_plot_motion(scores, segs, thr, out_path):
    if not HAS_MPL:
        return
    x = np.arange(len(scores))
    plt.figure()
    plt.plot(x, scores)
    plt.axhline(thr, linestyle="--")
    for (a, b) in segs:
        plt.axvspan(a, b, alpha=0.2)
    plt.title("Movimiento (score) vs frame")
    plt.xlabel("frame")
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


def run_on_video(video_path: Path, cfg: Config):
    print(f"\n=== {video_path.name} ===")
    scores, fps = compute_motion_scores(video_path, cfg)
    segs, thr = find_static_segments(scores, fps, cfg)

    print(f"FPS: {fps:.2f} | Frames: {len(scores)}")
    print(f"Umbral reposo (thr): {thr:.6f} | Segmentos detectados: {len(segs)} -> {segs}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.debug:
        maybe_plot_motion(scores, segs, thr, out_dir / f"{video_path.stem}_motion.png")

    if not segs:
        print("No se detectaron tramos de reposo con los parámetros actuales.")
        return

    # Elegimos el segmento más largo (más probable reposo real)
    seg = max(segs, key=lambda ab: ab[1] - ab[0])
    idx, vals, dbg = pick_best_frame(video_path, seg, cfg)

    if not vals:
        print("No se pudo estimar el valor de los dados en el segmento encontrado.")
        return

    # Orden final (5 valores)
    print(f"Frame elegido: {idx} | Valores (izq->der, arriba->abajo): {vals}")

    if cfg.debug and dbg is not None:
        annot = draw_debug(dbg["img"], dbg["boxes"], dbg["vals"])
        cv2.imwrite(str(out_dir / f"{video_path.stem}_frame_{idx}_annot.png"), annot)
        cv2.imwrite(str(out_dir / f"{video_path.stem}_frame_{idx}_dice_mask.png"), dbg["mask"])
        # nota: hue estimado
        with open(out_dir / f"{video_path.stem}_frame_{idx}_meta.txt", "w", encoding="utf-8") as f:
            f.write(f"hue_estimado={dbg['hue']}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None, help="Paths a tirada_<id>.mp4")
    ap.add_argument("--no-debug", action="store_true", help="No guardar imágenes/plots intermedios")
    ap.add_argument("--out", default="tp3_debug", help="Carpeta de salida para debug")
    args = ap.parse_args()

    cfg = Config(debug=not args.no_debug, out_dir=args.out)

    if args.videos:
        vids = [Path(v) for v in args.videos]
    else:
        # defaults típicos en el mismo directorio
        vids = [Path(f"TP3\\tirada_{i}.mp4") for i in [1, 2, 3, 4]]

    for vp in vids:
        if not vp.exists():
            print(f"[WARN] No existe: {vp}")
            continue
        run_on_video(vp, cfg)


if __name__ == "__main__":
    main()
