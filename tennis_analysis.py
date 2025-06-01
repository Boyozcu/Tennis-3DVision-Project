import cv2
import numpy as np
import time
import math

VIDEO_PATH = "data/tennis.mp4"
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 60

BLUE_COURT_HSV_LOWER = np.array([90, 20, 20])
BLUE_COURT_HSV_UPPER = np.array([140, 255, 255])
COURT_LINE_MIN_LENGTH = 50
COURT_LINE_MAX_GAP = 10
CANNY_THRESH_LOW = 50
CANNY_THRESH_HIGH = 150

SOURCE_POINTS = np.array([[419, 219], [863, 222], [1035, 575], [241, 570]], dtype=np.float32)

PLAYER_MIN_AREA = 800
PLAYER_MAX_AREA = 20000
PLAYER_MIN_ASPECT_RATIO = 1.2
PLAYER_MAX_ASPECT_RATIO = 4.5
PLAYER_MIN_SOLIDITY = 0.6
PLAYER_MIN_Y = 200
PLAYER_MAX_Y = 650
PLAYER_MIN_X = 100
PLAYER_MAX_X = 1180
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 16

BALL_HSV_LOWER = np.array([49, 31, 111])
BALL_HSV_UPPER = np.array([122, 127, 234])
BALL_MIN_AREA = 10
BALL_MAX_AREA = 300
BALL_MIN_CIRCULARITY = 0.4
BALL_MAX_SPEED = 150
BALL_SEARCH_WINDOW_MULTIPLIER = 1.5
BALL_MORPH_KERNEL_SIZE = 5

SKETCH_WIDTH = 400
SKETCH_HEIGHT = 600
COURT_SKETCH_COLOR = (180, 130, 70)

DESTINATION_POINTS = np.array([[100, 100], [300, 100], [300, 500], [100, 500]], dtype=np.float32)

SKETCH_LINES = [
    (DESTINATION_POINTS[0], DESTINATION_POINTS[1]),
    (DESTINATION_POINTS[1], DESTINATION_POINTS[2]),
    (DESTINATION_POINTS[2], DESTINATION_POINTS[3]),
    (DESTINATION_POINTS[3], DESTINATION_POINTS[0]),
    (np.array([DESTINATION_POINTS[0][0], DESTINATION_POINTS[0][1] + (DESTINATION_POINTS[3][1] - DESTINATION_POINTS[0][1]) * (1/3)], dtype=np.float32), np.array([DESTINATION_POINTS[1][0], DESTINATION_POINTS[1][1] + (DESTINATION_POINTS[2][1] - DESTINATION_POINTS[1][1]) * (1/3)], dtype=np.float32)),
    (np.array([DESTINATION_POINTS[0][0], DESTINATION_POINTS[0][1] + (DESTINATION_POINTS[3][1] - DESTINATION_POINTS[0][1]) * (2/3)], dtype=np.float32), np.array([DESTINATION_POINTS[1][0], DESTINATION_POINTS[1][1] + (DESTINATION_POINTS[2][1] - DESTINATION_POINTS[1][1]) * (2/3)], dtype=np.float32)),
    (np.array([DESTINATION_POINTS[0][0] + (DESTINATION_POINTS[1][0] - DESTINATION_POINTS[0][0]) / 2, DESTINATION_POINTS[0][1]], dtype=np.float32), np.array([DESTINATION_POINTS[3][0] + (DESTINATION_POINTS[2][0] - DESTINATION_POINTS[3][0]) / 2, DESTINATION_POINTS[3][1]], dtype=np.float32)),
    (np.array([DESTINATION_POINTS[0][0], DESTINATION_POINTS[0][1] + (DESTINATION_POINTS[3][1] - DESTINATION_POINTS[0][1]) / 2], dtype=np.float32), np.array([DESTINATION_POINTS[1][0], DESTINATION_POINTS[1][1] + (DESTINATION_POINTS[2][1] - DESTINATION_POINTS[1][1]) / 2], dtype=np.float32))
]

RALLY_END_FRAME_DIFF = 120
SHOT_CHANGE_DIRECTION_FRAME_DIFF = 30
NET_TO_BASELINE_METERS = 11.89
COURT_HALF_WIDTH_METERS = 8.23
COURT_ZONE_X_DIVISIONS = [100, 100 + (300-100)//3, 300 - (300-100)//3, 300]
COURT_ZONE_NAMES = ["Zone_1", "Zone_2", "Zone_3"]

class TennisAnalyzer:
    def __init__(self):
        self.cap = None
        self.out_main = None
        self.out_sketch = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD, detectShadows=False)
        self.homography_matrix = None
        self.kalman_filter = self.create_kalman_filter()
        self.last_ball_pos = None
        self.ball_positions = []
        self.player_trackers = []
        self.frame_count = 0
        self.rally_count = 0
        self.ball_direction_changes = 0
        self.ball_last_y_direction = None
        self.ball_last_detected_frame = 0
        self.zone_hits = {zone: 0 for zone in COURT_ZONE_NAMES}
        self.serve_speeds = []

    def create_kalman_filter(self):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        kalman.errorCovPost = 0.1 * np.eye(4, dtype=np.float32)
        return kalman

    def detect_court_lines(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=100, minLineLength=COURT_LINE_MIN_LENGTH, maxLineGap=COURT_LINE_MAX_GAP)
        
        if lines is not None and len(lines) > 4:
            try:
                self.homography_matrix = cv2.findHomography(SOURCE_POINTS, DESTINATION_POINTS)[0]
                return True
            except:
                return False
        return False

    def detect_players(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        _, binary_image = cv2.threshold(fg_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=4)
        
        valid_players = []
        for i in range(1, len(stats)):
            x, y, w, h, area = stats[i]
            
            if area < PLAYER_MIN_AREA or area > PLAYER_MAX_AREA:
                continue
                
            if x < PLAYER_MIN_X or x + w > PLAYER_MAX_X or y < PLAYER_MIN_Y or y + h > PLAYER_MAX_Y:
                continue
                
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio < PLAYER_MIN_ASPECT_RATIO or aspect_ratio > PLAYER_MAX_ASPECT_RATIO:
                continue
                
            w = max(w, 25)
            h = max(h, 40)
            valid_players.append((x, y, w, h, area))
        
        valid_players.sort(key=lambda x: x[4], reverse=True)
        
        final_players = []
        for i, player in enumerate(valid_players[:2]):
            if i == 1 and len(final_players) > 0:
                x1, y1, w1, h1 = final_players[0][:4]
                x2, y2, w2, h2 = player[:4]
                dist = np.sqrt(((x1+w1//2) - (x2+w2//2))**2 + ((y1+h1//2) - (y2+h2//2))**2)
                if dist < 150:
                    continue
            final_players.append(player)
        
        return final_players[:2]

    def detect_ball(self, frame, search_window=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
        
        if search_window:
            window_mask = np.zeros_like(mask)
            x, y, w, h = search_window
            window_mask[y:y+h, x:x+w] = 255
            mask = cv2.bitwise_and(mask, window_mask)
        
        kernel = np.ones((BALL_MORPH_KERNEL_SIZE, BALL_MORPH_KERNEL_SIZE), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < BALL_MIN_AREA or area > BALL_MAX_AREA:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < BALL_MIN_CIRCULARITY:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w//2, y + h//2
            
            if self.last_ball_pos:
                distance = np.sqrt((center_x - self.last_ball_pos[0])**2 + (center_y - self.last_ball_pos[1])**2)
                if distance > BALL_MAX_SPEED:
                    continue
                    
            score = circularity * area
            if score > best_score:
                best_score = score
                best_ball = (center_x, center_y)
        
        return best_ball

    def track_ball(self, frame):
        search_window = None
        if self.last_ball_pos:
            x, y = self.last_ball_pos
            window_size = int(BALL_MAX_SPEED * BALL_SEARCH_WINDOW_MULTIPLIER)
            search_window = (max(0, x - window_size//2), max(0, y - window_size//2), 
                           min(VIDEO_WIDTH, window_size), min(VIDEO_HEIGHT, window_size))
        
        ball_pos = self.detect_ball(frame, search_window)
        
        if not ball_pos and self.last_ball_pos:
            ball_pos = self.detect_ball(frame)
        
        if ball_pos:
            self.kalman_filter.correct(np.array([[np.float32(ball_pos[0])], [np.float32(ball_pos[1])]]))
            self.last_ball_pos = ball_pos
            self.ball_last_detected_frame = self.frame_count
            self.ball_positions.append((ball_pos, self.frame_count))
            
            if len(self.ball_positions) >= 2:
                curr_y = ball_pos[1]
                prev_y = self.ball_positions[-2][0][1]
                curr_direction = 1 if curr_y > prev_y else -1
                
                if self.ball_last_y_direction is not None and curr_direction != self.ball_last_y_direction:
                    if self.frame_count - self.ball_positions[-2][1] >= SHOT_CHANGE_DIRECTION_FRAME_DIFF:
                        self.ball_direction_changes += 1
                        
                self.ball_last_y_direction = curr_direction
        else:
            prediction = self.kalman_filter.predict()
            predicted_pos = (int(prediction[0, 0]), int(prediction[1, 0]))
            if (0 <= predicted_pos[0] < VIDEO_WIDTH and 0 <= predicted_pos[1] < VIDEO_HEIGHT):
                ball_pos = predicted_pos
        
        if self.frame_count - self.ball_last_detected_frame > RALLY_END_FRAME_DIFF:
            self.rally_count += 1
            self.ball_last_detected_frame = self.frame_count
        
        return ball_pos

    def map_to_bird_view(self, point):
        if self.homography_matrix is not None:
            try:
                point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(point_array, self.homography_matrix)
                return (int(transformed[0][0][0]), int(transformed[0][0][1]))
            except:
                pass
        return None

    def analyze_zone_hit(self, bird_point):
        if bird_point:
            x = bird_point[0]
            for i, zone_x in enumerate(COURT_ZONE_X_DIVISIONS[:-1]):
                if zone_x <= x < COURT_ZONE_X_DIVISIONS[i+1]:
                    zone_name = COURT_ZONE_NAMES[min(i, len(COURT_ZONE_NAMES)-1)]
                    self.zone_hits[zone_name] += 1
                    break

    def create_sketch_frame(self, players, ball_pos):
        sketch = np.full((SKETCH_HEIGHT, SKETCH_WIDTH, 3), COURT_SKETCH_COLOR, dtype=np.uint8)
        
        for line in SKETCH_LINES:
            start_pt = tuple(map(int, line[0]))
            end_pt = tuple(map(int, line[1]))
            cv2.line(sketch, start_pt, end_pt, (255, 255, 255), 2)
        
        for i, player in enumerate(players):
            bird_point = self.map_to_bird_view((player[0] + player[2]//2, player[1] + player[3]//2))
            if bird_point:
                color = (0, 0, 255) if i == 0 else (255, 0, 0)
                cv2.circle(sketch, bird_point, 8, color, -1)
                cv2.putText(sketch, f'P{i+1}', (bird_point[0]-10, bird_point[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if ball_pos:
            bird_ball = self.map_to_bird_view(ball_pos)
            if bird_ball:
                cv2.circle(sketch, bird_ball, 5, (0, 255, 255), -1)
                cv2.putText(sketch, 'Ball', (bird_ball[0]-15, bird_ball[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                self.analyze_zone_hit(bird_ball)
        
        return sketch

    def process_frame(self, frame):
        players = self.detect_players(frame)
        ball_pos = self.track_ball(frame)
        
        if self.frame_count % 30 == 0:
            court_detected = self.detect_court_lines(frame)
        else:
            court_detected = self.homography_matrix is not None
        
        status = "Detected" if len(players) == 2 and ball_pos and court_detected else "INVALID VIEW"
        
        output_frame = frame.copy()
        
        for i, player in enumerate(players):
            x, y, w, h = player[:4]
            color = (0, 0, 255) if i == 0 else (255, 0, 0)
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(output_frame, f'P{i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if ball_pos:
            cv2.circle(output_frame, ball_pos, 8, (0, 255, 255), -1)
            cv2.putText(output_frame, 'Ball', (ball_pos[0]-20, ball_pos[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(output_frame, f'Frame: {self.frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if status == "Detected" else (0, 0, 255), 2)
        
        sketch_frame = self.create_sketch_frame(players, ball_pos)
        
        return output_frame, sketch_frame, len(players) == 2 and ball_pos is not None

    def get_user_confirmation(self, frame1, sketch1, frame2, sketch2):
        print("\n=== BAŞLANGIÇ KONTROLÜ ===")
        print("İlk 2 başarılı tespit karesi görüntüleniyor...")
        print("Devam etmek için 'y' tuşuna basın, çıkmak için herhangi bir tuşa basın.")
        
        combined1 = np.hstack([frame1, cv2.resize(sketch1, (frame1.shape[1]//2, frame1.shape[0]))])
        combined2 = np.hstack([frame2, cv2.resize(sketch2, (frame2.shape[1]//2, frame2.shape[0]))])
        combined = np.vstack([combined1, combined2])
        
        cv2.imshow('Detection Preview - Frame 1 & 2', cv2.resize(combined, (1200, 800)))
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        return key == ord('y') or key == ord('Y')

    def run(self):
        try:
            self.cap = cv2.VideoCapture(VIDEO_PATH)
            if not self.cap.isOpened():
                print(f"Video dosyası açılamadı: {VIDEO_PATH}")
                return
            
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out_main = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
            self.out_sketch = cv2.VideoWriter('sketch_output.mp4', fourcc, fps, (SKETCH_WIDTH, SKETCH_HEIGHT))
            
            successful_frames = []
            while len(successful_frames) < 2:
                ret, frame = self.cap.read()
                if not ret:
                    print("Video okunurken hata oluştu!")
                    return
                
                output_frame, sketch_frame, success = self.process_frame(frame)
                self.frame_count += 1
                
                if success:
                    successful_frames.append((output_frame, sketch_frame))
            
            if not self.get_user_confirmation(successful_frames[0][0], successful_frames[0][1], 
                                            successful_frames[1][0], successful_frames[1][1]):
                print("Kullanıcı tarafından iptal edildi.")
                return
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            
            print("\nVideo işleme başlatılıyor...")
            start_time = time.time()
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                output_frame, sketch_frame, _ = self.process_frame(frame)
                
                self.out_main.write(output_frame)
                self.out_sketch.write(sketch_frame)
                
                self.frame_count += 1
                
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"İşlenen kare: {self.frame_count}, Geçen süre: {elapsed:.1f}s")
            
            print("\n=== OYUN İSTATİSTİKLERİ ===")
            print(f"Toplam Ralli Sayısı: {self.rally_count}")
            print(f"Top Yönü Değişimi: {self.ball_direction_changes}")
            print(f"İşlenen Toplam Kare: {self.frame_count}")
            
            print("\nKort Bölgesi Hedefleme:")
            for zone, hits in self.zone_hits.items():
                print(f"  {zone}: {hits} vuruş")
            
            print(f"\nVideo işleme tamamlandı! Çıktı dosyaları: output.mp4, sketch_output.mp4")
            
        except Exception as e:
            print(f"Hata oluştu: {e}")
        finally:
            if self.cap:
                self.cap.release()
            if self.out_main:
                self.out_main.release()
            if self.out_sketch:
                self.out_sketch.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = TennisAnalyzer()
    analyzer.run()
