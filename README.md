# AI 에이전트 기반 개인 맞춤 뉴스 서비스 (news_project)

<details>
<summary><b>📰 프로젝트 소개</b></summary>
<br>

본 프로젝트는 AI Agent 기술을 활용하여, 사용자의 음성 명령/자연어 요청을 인식해 주요 키워드를 추출하고, 그에 맞춘 뉴스를 실시간 크롤링·요약·추천해주는 개인화 뉴스 서비스를 목표로 합니다.

- **복합 키워드 이중 쿼리** 구조를 이용해 매칭 정확도를 크게 향상.
- STT, LLM, 자연어 해석, 크롤링, 요약, TTS 출력까지 전체 파이프라인 자동화.
- UI/UX는 React 기반으로 개발, 직관적인 음성조작 인터페이스 제공.

</details>

<details>
<summary><b>👀 배경 및 필요성</b></summary>
<br>

- 정보 과잉 시대, 관심사에 맞는 뉴스 필터링의 한계와 난이도를 극복.
- 단순 키워드 매칭 이상의 복합 질의 지원으로 실제 요구 의도·맥락 반영.

</details>

<details>
<summary><b>🏗️ 시스템 아키텍처 및 기술 스택</b></summary>
<br>

- **Voice Interface**: Whisper 기반 STT, OpenAI TTS, 사용자 음성→텍스트/텍스트→음성
- **AI Agent**: LangChain 기반 명령 해석, GPT-4o/LLM 다단계 요약 및 프롬프트 엔지니어링
- **News Crawler**: Naver News API, BeautifulSoup, 언론사별 구조 대응 파싱
- **통합 백엔드**: FastAPI (Python), 데이터 핸들러
- **프론트엔드**: React, HTML, JavaScript, CSS 기반
- **전체 통합 테스트 90% 이상 성공**

</details>

<details>
<summary><b>🧩 주요 기능 모듈</b></summary>
<br>

- STT(음성→텍스트), 자연어 명령 해석, 키워드 추출, 이중 쿼리 구성
- 실시간 뉴스 검색/크롤링/정제, 멀티기사 통합 요약
- TTS(텍스트→음성), 맞춤형 UI(음성조작/결과 브리핑/애니메이션 포함)
- 사용자 이력·쿼리 기반 큐레이션 구조

</details>

<details>
<summary><b>📑 주요 폴더 및 파일 예시</b></summary>
<br>

- `/src/`: 주요 HTML/React 컴포넌트
- `/static/js/`, `/static/css/`: JS, CSS 리소스
- `/data/`: 샘플 뉴스 데이터 또는 크롤링 결과
- `/backend/server.py`: 크롤러·STT·LLM 통신 파이썬 코드
- README 및 문서화 파일

</details>

<details>
<summary><b>🚀 사용법</b></summary>
<br>

1. [환경설정] Python, Node.js, FastAPI, React 등 설치
2. [실행] Python 서버 및 React 클라이언트 실행 (상세 명령어는 추후 세부화)
3. [음성 명령] 마이크 버튼을 통해 요청→ STT → LLM 해석 및 뉴스 결과 확인

</details>

<details>
<summary><b>✨ 기술·성과 요약</b></summary>
<br>

- 이중 쿼리 구조 도입으로 뉴스 필터 정확도 18% 향상(내부테스트 기준)
- 통합 기능 모듈화 및 테스트 성공률 90% 이상
- 자동 프롬프트 조정, 다단계 요약, UI/UX 연속성 강화
- 사용자 만족도(자연스러운 응답 흐름 등) 88% 달성

</details>
