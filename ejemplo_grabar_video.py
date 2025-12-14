import cv2

# --- Leer y grabar un video ------------------------------------------------
cap = cv2.VideoCapture('tirada_1.mp4')  # Abre el archivo de video especificado ('tirada_1.mp4') para su lectura.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Obtiene los cuadros por segundo (FPS) del video usando CAP_PROP_FPS.
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Obtiene el número total de frames en el video usando CAP_PROP_FRAME_COUNT.

out = cv2.VideoWriter('Video-Output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
# Crear un objeto para escribir el video de salida.
#   - 'Video-Output.mp4': Nombre del archivo de salida.
#   - cv2.VideoWriter_fourcc(*'mp4v'): Codec utilizado para el archivo de salida.
#   - fps: Cuadros por segundo del video de salida, debe coincidir con el video de entrada.
#   - (width, height): Dimensiones del frame de salida, deben coincidir con las dimensiones originales del video.

while (cap.isOpened()): # Verifica si el video se abrió correctamente.
    
    ret, frame = cap.read()  # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.

    if ret == True:

        cv2.rectangle(frame, (100,100), (200,200), (0,0,255), 2)

        frame_show = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.

        cv2.imshow('Frame', frame_show) # Muestra el frame redimensionado.

        out.write(frame)   # Escribe el frame original (sin redimensionar) en el archivo de salida 'Video-Output.mp4'. IMPORTANTE: El tamaño del frame debe coincidir con el tamaño especificado al crear 'out'.
        if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
            break
    else:
        break

cap.release() # Libera el objeto 'cap', cerrando el archivo.
out.release() # Libera el objeto 'out',  cerrando el archivo.
cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.
