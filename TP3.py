import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os



# ================================ EJERCICIO A ======================================




def capturar_movimiento(video_path, resize_w=540, diff_thr=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scores = []
    prev = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # achicar el frame para que el score sea más estable
        h, w = frame.shape[:2]
        if w > resize_w:
            scale = resize_w / float(w)
            frame = cv2.resize(frame, (resize_w, int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev is None:
            scores.append(1.0)
        else:
            diff = cv2.absdiff(gray, prev)
            changed = (diff > diff_thr).astype(np.uint8)
            scores.append(changed.mean())  # porcentaje de pixeles que cambiaron

        prev = gray

    cap.release()
    return np.array(scores, dtype=np.float32), float(fps)


def encontrar_estaticos(scores, fps, percentile=10, min_static_sec=0.30):
    # umbral adaptativo (no fijo): percentil bajo de los scores
    tail = scores[min(5, len(scores)):]  # ignoramos los primeros frames
    thr = float(np.percentile(tail, percentile))

    min_len = max(3, int(min_static_sec * fps))

    segs = []
    i, n = 0, len(scores)
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

    return segs, thr, min_len


def encontrar_frame(video_path, seg, pick="min_motion", scores=None):
    a, b = seg
    if pick == "middle" or scores is None:
        idx = (a + b) // 2
    else:
        idx = int(a + np.argmin(scores[a:b]))

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("No pude leer el frame elegido.")
    return idx, frame


def recortar_zona_verde(frame, top=0.05, bottom=0.80, left=0.03, right=0.97):
    H, W = frame.shape[:2]
    y0 = int(H * top)
    y1 = int(H * bottom)
    x0 = int(W * left)
    x1 = int(W * right)
    return frame[y0:y1, x0:x1], x0, y0


def detectar_dados(frame_bgr, out_dir, expected=5):
    # Recortar zona verde (ROI)
    roi, offx, offy = recortar_zona_verde(frame_bgr, top=0.05, bottom=0.80, left=0.03, right=0.97)

    H, W = roi.shape[:2]
    img_area = H * W

    # HSV y máscara de ROJO (HUE)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 80, 50), (179, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Contornos sobre la máscara
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Candidatos por forma/tamaño
    candidatos = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.001 * img_area:
            continue
        if area > 0.05 * img_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h + 1e-6)
        if ar < 0.75 or ar > 1.35:
            continue

        candidatos.append((area, (x, y, w, h)))

    # Tomar los 5 más grandes
    candidatos.sort(key=lambda t: t[0], reverse=True)
    boxes = [b for _, b in candidatos[:expected]]

    # Orden fijo
    boxes.sort(key=lambda b: (b[1], b[0]))

    # Dibujar + crops
    anotada_roi = roi.copy()
    anotada_full = frame_bgr.copy()
    dados = []

    for i, (x, y, w, h) in enumerate(boxes, start=1):
        pad = int(0.08 * max(w, h))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)

        crop = roi[y0:y1, x0:x1].copy()

        # coords ROI
        box_roi = (x0, y0, x1 - x0, y1 - y0)

        # coords GLOBAL (para dibujar en el frame original / video)
        gx0 = x0 + offx
        gy0 = y0 + offy
        gx1 = x1 + offx
        gy1 = y1 + offy
        box_global = (gx0, gy0, gx1 - gx0, gy1 - gy0)

        # dibujar en ROI (debug)
        cv2.rectangle(anotada_roi, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(anotada_roi, f"Dado {i}", (x0, max(0, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # dibujar en frame completo (para debug y para el punto b)
        cv2.rectangle(anotada_full, (gx0, gy0), (gx1, gy1), (0, 255, 0), 2)
        cv2.putText(anotada_full, f"Dado {i}", (gx0, max(0, gy0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imwrite(out_dir + f"/dado_{i}.png", crop)

        dados.append({
            "id": i,
            "crop": crop,
            "box_roi": box_roi,
            "box": box_global,     # <-- ESTE es el que usás para dibujar en el video
            "offx": offx,
            "offy": offy
        })

    cv2.imwrite(out_dir + "/dados_boxes_roi.png", anotada_roi)
    cv2.imwrite(out_dir + "/dados_boxes_full.png", anotada_full)

    return dados



def save_motion_plot(scores, segs, thr, out_path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return  # si no hay matplotlib, no plotteamos

    x = np.arange(len(scores))
    plt.figure()
    plt.plot(x, scores)
    plt.axhline(thr, linestyle="--")
    for (a, b) in segs:
        plt.axvspan(a, b, alpha=0.2)
    plt.title("Movimiento (score) vs frame")
    plt.xlabel("frame")
    plt.ylabel("score (% pixeles cambiados)")
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()




# =================== CONTEO DE PIPS (con subcarpeta conteo_pips) ==================== 


def _roundness(contour):
    area = cv2.contourArea(contour)
    (xc, yc), r = cv2.minEnclosingCircle(contour)
    if r <= 0:
        return 0.0
    area_circ = np.pi * (r ** 2)
    return float(area / (area_circ + 1e-6))

def _aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return 999.0
    return max(w, h) / float(min(w, h))

def _resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)

def _montaje_5(imgs, target_h=280):
    # arma 1 imagen con 5 paneles en fila
    imgs2 = []
    for im in imgs:
        if im is None:
            im = np.zeros((target_h, target_h, 3), dtype=np.uint8)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        im = _resize_to_height(im, target_h)
        imgs2.append(im)
    return cv2.hconcat(imgs2)



def contar_pips_y_guardar(dados, out_dir):
    # En vez de guardar mil intermedios, guardamos 1 sola imagen final por tirada:
    # out_dir/pips_plot.png  (5 dados con contornos + valor)

    valores_dados = []
    paneles = []

    # Parámetros (tocables) para pips en HSV
    # pips = brillantes (V alto) y poco saturados (S bajo)
    PIP_V_MIN = 150
    PIP_S_MAX = 110

    # Filtros geométricos
    UMBRAL_ROUNDNESS_PIP = 0.60   # antes 0.70 (era muy estricto)
    UMBRAL_ASPECT_RATIO = 1.60    # antes 1.35 (era muy estricto)

    for d in dados:
        idx = d["id"]
        crop = d["crop"]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # máscara del dado (rojo) para limitar la zona
        mask1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 80, 50), (179, 255, 255))
        dice_mask = cv2.bitwise_or(mask1, mask2)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dice_mask = cv2.morphologyEx(dice_mask, cv2.MORPH_CLOSE, k, iterations=2)

        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dice_mask = cv2.dilate(dice_mask, k_dil, iterations=1)


        # --- DETECCIÓN DE PIPS (más robusta que Otsu) ---
        # pips: baja saturación + alto valor, pero solo dentro del dado
        pip_mask = cv2.inRange(hsv, (0, 0, PIP_V_MIN), (179, PIP_S_MAX, 255))
        pip_mask = cv2.bitwise_and(pip_mask, pip_mask, mask=dice_mask)

        # limpiar (sacar ruido, unir pips si quedaron rotos)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pip_mask = cv2.morphologyEx(pip_mask, cv2.MORPH_OPEN, k3, iterations=1)
        pip_mask = cv2.morphologyEx(pip_mask, cv2.MORPH_CLOSE, k3, iterations=1)



        # contornos pips
        cont_pips, _ = cv2.findContours(pip_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        roi_area = pip_mask.shape[0] * pip_mask.shape[1]
        PIP_AREA_MIN = 0.0005 * roi_area   # más permisivo
        PIP_AREA_MAX = 0.02   * roi_area   # más estricto arriba

        validos = []
        for c in cont_pips:
            area = cv2.contourArea(c)
            if not (PIP_AREA_MIN < area < PIP_AREA_MAX):
                continue

            r = _roundness(c)
            ar = _aspect_ratio(c)

            if (r >= UMBRAL_ROUNDNESS_PIP) and (ar <= UMBRAL_ASPECT_RATIO):
                validos.append(c)

        num_pips = len(validos)
        valores_dados.append(num_pips)
        d["valor"] = num_pips

        # Panel visual: crop + contornos + texto
        panel = crop.copy()
        cv2.drawContours(panel, validos, -1, (0, 255, 0), 2)

        paneles.append(panel)

    # guardamos un solo “plot” por tirada
    plot_img = _montaje_5(paneles, target_h=280)
    cv2.imwrite(out_dir + "/pips_plot_" + os.path.basename(out_dir) + ".png", plot_img)

    return valores_dados




#================================ PRUEBA ========================================


if __name__ == "__main__":
    BASE = str(Path(__file__).resolve().parent)

    base_debug = BASE + "/debug_reposo"
    if not os.path.exists(base_debug):
        os.makedirs(base_debug)

    for i in [1, 2, 3, 4]:
        video = BASE + f"/tirada_{i}.mp4"
        out_dir = base_debug + f"/tirada_{i}"

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not os.path.exists(video):
            print(f"[WARN] No existe el video: {video}")
            continue

        # 1) motion score
        scores, fps = capturar_movimiento(video)

        # 2) reposos
        segs, thr, min_len = encontrar_estaticos(scores, fps)

        # fallback simple si no hay reposo
        if len(segs) == 0:
            segs, thr, min_len = encontrar_estaticos(scores, fps, percentile=20)
        if len(segs) == 0:
            segs, thr, min_len = encontrar_estaticos(scores, fps, percentile=30)

        print("\n==============================")
        print("VIDEO:", f"tirada_{i}.mp4")
        print("OUT:", out_dir)
        print("FPS:", fps)
        print("thr:", thr, "| min_len:", min_len)
        print("segmentos:", segs)

        if len(segs) == 0:
            print("No se detectó reposo.")
            continue

        # Motion Plot (siempre)
        save_motion_plot(scores, segs, thr, out_dir + "/motion_plot.png")
        print("Guardado: motion_plot.png")

        # Probar cada reposo y quedarnos con el primero que detecte 5 dados
        encontrado = False
        dados_ok = None
        frame_ok = None
        idx_ok = None
        seg_ok = None

        for seg in segs:
            start, end = seg
            idx = (start + end) // 2

            cap = cv2.VideoCapture(video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            cap.release()

            if not ok:
                continue

            dados = detectar_dados(frame, out_dir, expected=5)

            if len(dados) == 5:
                encontrado = True
                dados_ok = dados
                frame_ok = frame
                idx_ok = idx
                seg_ok = seg
                break

        if not encontrado:
            print("ERROR: no se encontró ningún reposo con 5 dados")
            continue

        # Guardar frame reposo elegido
        cv2.imwrite(str(out_dir + "/frame_reposo.png"), frame_ok)
        print("Frame reposo elegido:", idx_ok, "| segmento:", seg_ok)
        print("Guardado: frame_reposo.png")

        # Conteo de pips + guardar intermedios en subcarpeta
        valores = contar_pips_y_guardar(dados_ok, out_dir)

        # Imprimir resultado formateado
        print(f"\nTIRADA {i}:")
        for j, v in enumerate(valores, start=1):
            print(f"Dado {j}: {v} pips detectados")




# ================================ EJERCICIO B ======================================


# ===== Conteo de pips (SIN guardar subcarpetas) =====


def contar_pips(dados):
    # misma idea que venías usando: pips = baja saturación + alto valor (HSV),
    # dentro del dado rojo

    valores = []

    PIP_V_MIN = 150
    PIP_S_MAX = 110

    UMBRAL_ROUNDNESS_PIP = 0.60
    UMBRAL_ASPECT_RATIO = 1.60

    for d in dados:
        crop = d["crop"]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # máscara del dado rojo
        mask1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 80, 50), (179, 255, 255))
        dice_mask = cv2.bitwise_or(mask1, mask2)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dice_mask = cv2.morphologyEx(dice_mask, cv2.MORPH_CLOSE, k, iterations=2)

        # (fix típico del pip perdido)
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dice_mask = cv2.dilate(dice_mask, k_dil, iterations=1)

        # máscara de pips
        pip_mask = cv2.inRange(hsv, (0, 0, PIP_V_MIN), (179, PIP_S_MAX, 255))
        pip_mask = cv2.bitwise_and(pip_mask, pip_mask, mask=dice_mask)

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pip_mask = cv2.morphologyEx(pip_mask, cv2.MORPH_OPEN, k3, iterations=1)
        pip_mask = cv2.morphologyEx(pip_mask, cv2.MORPH_CLOSE, k3, iterations=1)

        cont_pips, _ = cv2.findContours(pip_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        roi_area = pip_mask.shape[0] * pip_mask.shape[1]
        PIP_AREA_MIN = 0.0005 * roi_area
        PIP_AREA_MAX = 0.02   * roi_area

        validos = []
        for c in cont_pips:
            area = cv2.contourArea(c)
            if not (PIP_AREA_MIN < area < PIP_AREA_MAX):
                continue

            r = _roundness(c)
            ar = _aspect_ratio(c)

            if (r >= UMBRAL_ROUNDNESS_PIP) and (ar <= UMBRAL_ASPECT_RATIO):
                validos.append(c)

        num = len(validos)
        d["valor"] = num
        valores.append(num)

    return valores


# ===== Utilidad: ver si frame está en un segmento =====

def buscar_segmento(frame_idx, segmentos_anotados, pad=15):
    for s in segmentos_anotados:
        a, b = s["seg"]
        a2 = max(0, a - pad)
        b2 = b + pad
        if a2 <= frame_idx < b2:
            return s
    return None

# ===== Ejercicio B: generar video anotado =====

def generar_video_anotado(video_path, out_video_path, out_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir:", video_path)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 1) reposos (con tu lógica existente)
    scores, fps_scores = capturar_movimiento(video_path)  # fps_scores debería coincidir con fps
    segs, thr, min_len = encontrar_estaticos(scores, fps_scores)

    if len(segs) == 0:
        segs, thr, min_len = encontrar_estaticos(scores, fps_scores, percentile=20)
    if len(segs) == 0:
        segs, thr, min_len = encontrar_estaticos(scores, fps_scores, percentile=30)

    # guardamos plot como debug (sirve para el informe)
    save_motion_plot(scores, segs, thr, out_dir + "/motion_plot.png")

    # 2) Precomputar anotaciones por segmento
    segmentos_anotados = []

    for seg in segs:
        start, end = seg
        idx = (start + end) // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        dados = detectar_dados(frame, out_dir, expected=5)
        if len(dados) != 5:
            continue

        valores = contar_pips(dados)

        # guardamos esta “foto” de anotación para usarla en todo el segmento
        segmentos_anotados.append({
            "seg": seg,
            "dados": dados,      # incluye box global + id + valor
            "valores": valores
        })

    # 3) Preparar writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # volver al inicio
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        seg_act = buscar_segmento(frame_idx, segmentos_anotados, pad=5) # podriamos aumentar el pad para que deje los BB mas tiempo

        if seg_act is not None:
            # dibujar solo si estamos en reposo anotado
            for d in seg_act["dados"]:
                x, y, w, h = d["box"]
                valor = d.get("valor", "?")
                did = d.get("id", "?")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Dado {did}: {valor}",
                            (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


# ===== MAIN B: uno por cada archivo =====

if __name__ == "__main__":
    BASE = str(Path(__file__).resolve().parent)
    base_debug = BASE + "/debug_reposo"
    os.makedirs(base_debug, exist_ok=True)

    for i in [1, 2, 3, 4]:
        video = BASE + f"/tirada_{i}.mp4"
        out_dir = base_debug + f"/tirada_{i}"
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(video):
            print(f"[WARN] No existe el video: {video}")
            continue

        out_video = out_dir + f"/tirada_{i}_anotada.mp4"
        generar_video_anotado(video, out_video, out_dir)

        print(f"Listo: {out_video}")
