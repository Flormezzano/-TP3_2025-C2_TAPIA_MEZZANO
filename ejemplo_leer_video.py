import os
import cv2

os.makedirs("frames", exist_ok = True)  # Si no existe, crea la carpeta 'frames' en el directorio actual.

# --- Leer un video ------------------------------------------------
cap = cv2.VideoCapture('tirada_1.mp4')  # Abre el archivo de video especificado ('tirada_1.mp4') para su lectura.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Obtiene los cuadros por segundo (FPS) del video usando CAP_PROP_FPS.
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Obtiene el número total de frames en el video usando CAP_PROP_FRAME_COUNT.

frame_number = 0
while (cap.isOpened()): # Verifica si el video se abrió correctamente.

    ret, frame = cap.read() # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.

    if ret == True:  

        frame = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.

        cv2.imshow('Frame', frame) # Muestra el frame redimensionado.

        # cv2.imwrite(os.path.join("frames", f"frame_{frame_number}.jpg"), frame) # Guarda el frame en el archivo './frames/frame_{frame_number}.jpg'.

        frame_number += 1
        if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
            break
    else:  
        break  

cap.release() # Libera el objeto 'cap', cerrando el archivo.
cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.
