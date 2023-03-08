import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No se puede capturar el video. Saliendo...")
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                # Extraer coordenadas de los dedos de la mano
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Calcular la distancia entre los dedos
                thumb_index_distance = ((thumb_tip.x - index_tip.x) * 2 + (thumb_tip.y - index_tip.y) * 2) ** 0.5
                index_middle_distance = ((index_tip.x - middle_tip.x) * 2 + (index_tip.y - middle_tip.y) * 2) ** 0.5

                # Realizar suma si la distancia entre el pulgar e índice y entre el índice y medio es menor a un valor umbral
                threshold = 0.03
                
        cv2.imshow("Suma de dedos", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()