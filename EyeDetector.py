import cv2
import dlib
import numpy as np
from dataclasses import dataclass

@dataclass
class EyeDetector:
    shape_predictor_path: str

    def detect_eye(self, image):
        # 이미지 크기 조정
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)

        # 얼굴이 검출되지 않은 경우
        if len(faces) == 0:
            return None

        # 랜드마크 검출
        predictor = dlib.shape_predictor(self.shape_predictor_path)
        landmarks = predictor(gray, faces[0])

        # 왼쪽 눈과 오른쪽 눈의 좌표를 추출
        left_eye = landmarks.parts()[36:42]
        right_eye = landmarks.parts()[42:48]

        # 좌표를 numpy 배열로 변환
        left_eye = np.array([[point.x, point.y] for point in left_eye])
        right_eye = np.array([[point.x, point.y] for point in right_eye])

        # 눈 이미지 추출
        left_eye_image = self.crop_eye_image(image, left_eye)
        right_eye_image = self.crop_eye_image(image, right_eye)

        return left_eye_image, right_eye_image

    def crop_eye_image(self, image, eye_points):
        # 눈의 중심 좌표 계산
        eye_center = np.mean(eye_points, axis=0)

        # 눈의 가로, 세로 길이 계산
        eye_width = np.abs(eye_points[0][0] - eye_points[3][0])
        eye_height = eye_width * 3 / 4

        # 눈의 각도 계산
        dx = eye_points[3][0] - eye_points[0][0]
        dy = eye_points[3][1] - eye_points[0][1]
        angle = np.degrees(np.arctan2(dy, dx))

        # 눈 중심을 기준으로 회전 변환 행렬 생성
        rotation_matrix = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale=1.0)

        # 이미지 회전 및 자르기
        image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        x = int(eye_center[0] - eye_width / 2)
        y = int(eye_center[1] - eye_height / 2)
        cropped_image = image[y:y+int(eye_height), x:x+int(eye_width)]

        return cropped_image