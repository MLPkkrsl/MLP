from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import random
from ultralytics import YOLO
import requests

app = Flask(__name__)

# YOLO 모델 경로 설정
MODEL_PATH = r"C:\Users\kj100\OneDrive\바탕 화면\MLP\find_wrong\asl_yolov8_model.pt"

# YOLO 모델 로드
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print("YOLO 모델이 성공적으로 로드되었습니다.")
else:
    raise FileNotFoundError(f"YOLO 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

# 알파벳 클래스 정의
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# 전역 변수
target_alphabet = None  # 학습 모드에서 사용자가 설정한 목표 알파벳
current_char = random.choice(classes)  # 퀴즈 모드에서 랜덤 알파벳
wrong_attempts = []  # 퀴즈에서 오답 기록
quiz_score = 0  # 퀴즈 점수
total_questions = 10  # 총 퀴즈 문제 수
current_question = 1  # 현재 퀴즈 문제 번호

# Google Gemini API 키
API_KEY = "AIzaSyDaU67xjmxRCO-kc8niC3Reb5OprzOlUJk"  # 구글 Gemini API 키 입력

def get_alphabet_description(alphabet):
    """Google Gemini API를 사용해 알파벳 설명 가져오기"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": f"알파벳 {alphabet}에 대해 설명해 주세요."}]}]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        content = response.json()
        explanation = content['candidates'][0]['content']['parts'][0]['text']
        return explanation
    else:
        return "설명을 가져올 수 없습니다."

def generate_api_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    global target_alphabet

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전

        # YOLO 예측
        results = model.predict(source=frame, conf=0.5, show=False, verbose=False)
        detected_alphabet = None

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if confidence > 0.5:
                    detected_alphabet = classes[class_id]

                    # 좌표 가져오기
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 색상: 초록(정답), 빨강(오답)
                    color = (0, 255, 0) if detected_alphabet == target_alphabet else (0, 0, 255)

                    # 사각형 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # 감지된 알파벳 텍스트 추가
                    cv2.putText(frame, f"{detected_alphabet} ({confidence:.2f})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 메시지 출력
        message = f"Correct: {detected_alphabet}" if detected_alphabet == target_alphabet else f"Detected: {detected_alphabet}" if detected_alphabet else "No sign detected"
        color = (0, 255, 0) if detected_alphabet == target_alphabet else (0, 0, 255)
        cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 프레임 출력
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def generate_learning_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    global target_alphabet

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전

        # YOLO 예측
        results = model.predict(source=frame, conf=0.5, show=False, verbose=False)
        detected_alphabet = None

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if confidence > 0.5:
                    detected_alphabet = classes[class_id]

                    # 좌표 가져오기
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 색상: 초록(정답), 빨강(오답)
                    color = (0, 255, 0) if detected_alphabet == target_alphabet else (0, 0, 255)

                    # 사각형 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # 감지된 알파벳 텍스트 추가
                    cv2.putText(frame, f"{detected_alphabet} ({confidence:.2f})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 상태 메시지 출력
        if detected_alphabet:
            message = "Correct" if detected_alphabet == target_alphabet else f"Detected: {detected_alphabet}"
        else:
            message = "No sign detected"

        # 상태 메시지 출력
        color = (0, 255, 0) if detected_alphabet == target_alphabet else (0, 0, 255)
        cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 프레임 출력
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

current_target_alphabet = None  # 전역 변수로 설정

def generate_learning_frames():
    """카메라 프레임 생성 및 YOLO 예측"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 초기 상태 설정
    global current_target_alphabet  # 필요한 경우 전역 변수로 선언
    current_target_alphabet = None  # 처음에는 아무 알파벳도 지정되지 않음

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전

        # YOLO 모델로 예측
        results = model.predict(source=frame, conf=0.5, show=False, verbose=False)
        detected_alphabet = None

        # YOLO 결과 처리
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if confidence > 0.5:
                    detected_alphabet = classes[class_id]

                    # 바운딩 박스 그리기
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{detected_alphabet} ({confidence:.2f})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 데이터 정규화: 공백 제거 및 대소문자 변환
        detected_alphabet = detected_alphabet.strip().upper() if detected_alphabet else None

        # 디버깅용 알파벳 출력
        print(f"Detected: {detected_alphabet}, Target: {current_target_alphabet}")

        # 정답 여부 확인
        if current_target_alphabet is None:
            result_message = "No target set"
            result_color = (255, 255, 0)
        elif detected_alphabet is None:
            result_message = "No sign detected"
            result_color = (0, 0, 255)
        elif detected_alphabet == current_target_alphabet:
            result_message = "Good Job!"
            result_color = (0, 255, 0)
        else:
            result_message = "Try Again!"
            result_color = (0, 0, 255)

        # 결과 메시지 출력
        cv2.putText(frame, result_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)

        # 프레임 출력
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    cap.release()
@app.route('/next_alphabet_quiz', methods=['POST'])
def next_alphabet_quiz():
    """퀴즈 모드에서 다음 알파벳으로 전환"""
    global current_char, current_question
    if current_question < total_questions:  # 퀴즈 질문 제한 확인
        current_char = random.choice(classes)  # 랜덤 알파벳 선택
        current_question += 1  # 질문 개수 증가
        return jsonify({'success': True, 'alphabet': current_char, 'question': current_question})
    else:
        return jsonify({'success': False, 'message': '퀴즈가 종료되었습니다.'})


def generate_quiz_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    global quiz_score, current_question, current_char

    while current_question <= total_questions:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전

        # YOLO 예측
        results = model.predict(source=frame, conf=0.5, show=False, verbose=False)
        detected_alphabet = None
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                if confidence > 0.5:
                    detected_alphabet = classes[class_id]

        # 정답 확인
        if detected_alphabet == current_char:
            quiz_score += 1
            current_question += 1
            current_char = random.choice(classes)  # 정답일 때 다음 알파벳
            print(f"정답! 다음 타겟 알파벳: {current_char}")
        elif detected_alphabet:
            # 오답 처리
            wrong_attempts.append((current_char, detected_alphabet))
            print(f"오답! 감지된 알파벳: {detected_alphabet}, 현재 타겟: {current_char}")

        # 'n' 키를 누르면 다음 알파벳으로 변경
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):  # 'n' 키가 입력되었을 경우
            current_char = random.choice(classes)  # 새로운 랜덤 알파벳 선택
            print(f"'n' 키 입력: 다음 타겟 알파벳으로 이동 - {current_char}")

        # 화면 표시
        color = (0, 255, 0) if detected_alphabet == current_char else (0, 0, 255)
        message = f"Detected: {detected_alphabet}" if detected_alphabet else "No sign detected"
        cv2.putText(frame, f"Target: {current_char}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Score: {quiz_score}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Question: {current_question}/{total_questions}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 프레임 출력
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



@app.route("/video_feed_wrongnote")
def video_feed_wrongnote():
    """오답 노트 모드 비디오 스트리밍"""
    return Response(generate_wrongnote_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_wrongnote_frames():
    global target_alphabet

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전

        # YOLO 예측
        results = model.predict(source=frame, conf=0.5, show=False, verbose=False)
        detected_alphabet = None
        confidence = 0

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                detected_alphabet = classes[class_id]

                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 사각형 색상
                box_color = (0, 255, 0) if detected_alphabet == target_alphabet else (0, 0, 255)

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # 알파벳과 정확도 표시
                cv2.putText(frame, f"{detected_alphabet} ({confidence:.2f})", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # 상태 메시지 (Correct / Incorrect)
        status_color = (0, 255, 0) if detected_alphabet == target_alphabet else (0, 0, 255)
        status_message = "Correct" if detected_alphabet == target_alphabet else "Incorrect"
        cv2.putText(frame, status_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # 프레임 출력
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route("/learning")
def learning_mode():
    """랜덤 알파벳 학습 모드 페이지"""
    sign_images_path = os.path.join(app.static_folder, "sign_images")
    
    # sign_images 폴더에서 모든 PNG 파일 목록 가져오기
    available_images = [f for f in os.listdir(sign_images_path) if f.endswith(".png")]
    
    if not available_images:
        alphabet_image = "default.png"  # 기본 이미지 처리 (default.png 파일 필요)
        alphabet_name = "None"  # 표시할 알파벳 이름
    else:
        # 랜덤 알파벳 선택
        alphabet_image = random.choice(available_images)
        alphabet_name = os.path.splitext(alphabet_image)[0]  # 파일 이름에서 확장자 제거

    # 랜덤으로 선택된 알파벳을 전달
    return render_template(
        "learning.html",
        alphabet_image=alphabet_image,
        alphabet_name=alphabet_name
    )


@app.route("/next_alphabet", methods=["POST"])
def next_alphabet():
    global current_target_alphabet
    # 랜덤 알파벳 선택 (예: A-Z 중 랜덤)
    import random
    alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    current_target_alphabet = random.choice(alphabets)
    alphabet_image = f"sign_images/{current_target_alphabet}.png"
    return jsonify({"success": True, "alphabet_name": current_target_alphabet, "alphabet_image": alphabet_image})

@app.route('/get_description', methods=['POST'])
def get_description():
    data = request.get_json()
    alphabet = data.get('alphabet', None)
    if not alphabet:
        return jsonify({"description": "Invalid input. Alphabet is required."}), 400
    try:
        description = get_alphabet_description(alphabet)
        return jsonify({"description": description})
    except Exception as e:
        print(f"Error fetching description: {e}")
        return jsonify({"description": "Failed to fetch description."}), 500


# 라우트 설정
@app.route("/")
def main_page():
    """메인 페이지"""
    return render_template("main.html")

@app.route("/quiz")
def quiz_mode():
    """퀴즈 모드 페이지"""
    return render_template("quiz.html")

@app.route("/api")
def api_mode():
    """API 모드 페이지"""
    return render_template("api.html")

@app.route("/video_feed_quiz")
def video_feed_quiz():
    """퀴즈 모드 비디오 스트리밍"""
    return Response(generate_quiz_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_feed_api")
def video_feed_api():
    """API 모드 비디오 스트리밍"""
    return Response(generate_api_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_feed_learning")
def video_feed_learning():
    """학습 모드 비디오 스트리밍"""
    return Response(generate_learning_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/set_alphabet", methods=["POST"])
def set_alphabet():
    """목표 알파벳 설정"""
    global target_alphabet
    target_alphabet = request.form.get("alphabet").upper()
    print(f"설정된 알파벳: {target_alphabet}")  # 로그 출력
    return {"message": f"목표 알파벳이 {target_alphabet}로 설정되었습니다.", "target": target_alphabet}

@app.route("/wrong_notes")
def wrong_notes():
    """오답 노트 페이지"""
    return render_template("wrong_notes.html", wrong_attempts=wrong_attempts)

if __name__ == "__main__":
    app.run(debug=True)
