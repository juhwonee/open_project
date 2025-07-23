# 2024 2학기 open_project 
## Member : MinsooKang, JangInhwanm, JeongJuHwon

# 🖐️ Hand Gesture-based Slide Presentation Controller

## 📌 선정 이유

기존 프레젠테이션은 리모컨이나 키보드 입력에 의존하여 슬라이드를 제어해야 하는 불편함이 있습니다.  
손 제스처는 **직관적이고 자연스러운 방식**으로 슬라이드를 제어할 수 있어 발표자의 집중력을 높이고,  
청중과의 **비언어적 소통**에도 도움이 됩니다.  

👉 본 프로젝트는 이러한 불편함을 해소하고  
**사용자 친화적인 인터페이스**의 가능성을 탐구하고자 기획되었습니다.

---

## 🧩 기술 스택 및 도구

| 분류 | 도구/프레임워크 |
|------|----------------|
| 핸드 트래킹 & 제스처 인식 | `MediaPipe`, `OpenCV`, `TensorFlow`, `YOLOv8n` |
| 슬라이드 제어 | `pyautogui`, `pynput` |
| 언어 | `Python` |
| 하드웨어 | 📷 웹캠 |

---

## 🔁 전체 동작 흐름
- **실시간 웹캠 영상 캡처**
- 인식된 제스처에 따라 슬라이드 넘김, 되돌리기 등 이벤트 매핑
- PyAutoGUI와 Pynput을 사용하여 키보드 입력을 모방해 슬라이드 제어

---

## 🚀 프로젝트 로드맵

- [x] 핸드 트래킹 및 제스처 인식 기능 구현
- [x] 인식된 제스처를 슬라이드 제어 동작으로 매핑
- [ ] 사용자 친화적인 UI와 통합
- [ ] 다양한 제스처 커스터마이징 지원

---

## 🌐 참고 오픈소스

- [`HandTrack.js`](https://github.com/victordibia/handtrack.js) – (JavaScript 기반)
- [`AI Virtual Mouse`](https://github.com/srbcheema1/AI-Virtual-Mouse) – MediaPipe 기반 마우스 제어
- [`Gesture Slide Control`](https://github.com/yourlink) – 슬라이드 제스처 컨트롤 예시 (직접 추가 가능)

---

## 🧠 기대 효과

- 발표자가 **시선이나 손의 위치를 바꾸지 않고** 슬라이드를 제어할 수 있음  
- 물리적 리모컨이 필요 없어 **무선 및 비접촉 제어 가능**  
- **AI 기반 사용자 인터페이스** 설계 경험 습득

---

> Made with ❤️ by Minsoo Kang  
> For presentation, interaction, and beyond.
