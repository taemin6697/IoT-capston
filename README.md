# VLM-based tip calculator to improve customer complaints
>tip calculators는 VLM 기반의 자동화된 팁 산정 서비스입니다. 카메라 영상과 사용자 리뷰 데이터를 종합 분석하여 서비스 품질을 객관적으로 평가하고, 이에 기반한 적절한 팁 금액을 제공합니다.
VLM의 강력한 멀티모달 인식 능력과 fine-tuning된 서비스 평가 기준을 활용해, 다양한 상황에 유연하게 대응하며 팁 산출의 일관성과 공정성을 높입니다. 또한, 행동 인식 기반 AI 모델을 통해 웨이터의 서빙 타이밍과 서비스 태도를 분석하고, 리뷰 분석 결과를 함께 반영하여 고객 만족도를 정량화합니다.
사용자는 직관적인 인터페이스를 통해 분석 결과와 추천 팁 금액을 손쉽게 확인할 수 있으며, 과도하거나 불공정한 팁 요구로 인한 부담을 줄이고 공정한 보상을 가능하게 합니다.

<br>

## Project BackGround

### 기획 의도 및 기대효과 


* **`배경`** : 최근 미국을 비롯한 팁 문화가 존재하는 국가들에서는 팁 문화가 사회적 갈등의 원인으로 떠오르고 있습니다. 기대에 미치지 못하는 서비스를 받았음에도 팁을 지불해야 하는 고객의 불만, 반대로 최상의 서비스를 제공했음에도 충분한 보상을 받지 못하는 종업원의 좌절이 반복되고 있습니다.

* **`목표`**: 저희는 이러한 갈등을 완화하고자, 웨이터의 서비스 행동과 식당에 대한 리뷰 데이터를 기반으로 공정하고 적절한 팁 금액을 산정하는 시스템을 개발하였습니다.

<br/>

### 기존 Legal Tech 서비스와의 차별점

- 기존의 팁 계산 서비스는 **고객이 직접 팁 비율을 입력해야 전체 금액을 산출하는 방식**이 일반적입니다. 반면, 본 시스템은 AI 모델을 활용하여 고객이 별도로 팁 비율을 설정하지 않아도, **서비스 품질에 따라 적절한 팁 비율을 자동으로 제안**한다는 점에서 차별화됩니다.


<br/>

## ⚙️ Use Case

<table>
  <tr>
    <td><img src="./image/Gradio.jpeg" alt="Image 2"/></td>
  </tr>
</table>

>1. 웹 서버 접속
>2. 메시지 프롬프트 창에 식당의 음식, 서비스에 대한 리뷰 및 별점을 입력
>3. AI 모델이 입력된 리뷰, 별점, 비디오 데이터, 그리고 해당 식당의 구글 리뷰를 종합 분석하여 적절한 팁 비율을 산출
>4. 산출된 팁 비율에 대한 AI의 분석 근거와 설명이 함께 제공
>5. 고객이 주문한 음식 금액과 팁 비율을 기반으로 최종 결제 금액이 자동 계산

<br>

## 🧑🏻‍💻 Team Introduction & Members 
> Team name : 두잔해
### 팀 소개
**한성대학교 지능시스템 트랙에서 모인, 캡스톤 디자인 팀 `두잔해`를 소개합니다!**

**우리는 정해진 역할보다 문제 해결에 집중합니다. 일이 보이면 먼저 손을 들고, 팀이 흔들릴 때는 중심을 잡습니다. 각자의 자리에서 주도적으로 움직이고, 끝까지 책임지는 자세가 우리 팀의 기본입니다.**

### 👨🏼‍💻 Members
김태민|고혜정|서준혁
:-:|:-:|:-:|
<img src='https://avatars.githubusercontent.com/u/96530685?v=4' height=130 width=130></img>|<img src='https://avatars.githubusercontent.com/u/190566247?v=4' height=130 width=130></img>|<img src='https://avatars.githubusercontent.com/u/105350096?v=4' height=130 width=130></img>|
<a href="https://github.com/taemin6697" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/Kohyejung" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/SeoBuAs" target="_blank"><img src="https://img.shields.io/badge/Github-black.svg?&style=round&logo=github"/></a>
<a href="mailto:taemin6697@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:helenko7738@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:withop9974@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|

<br>

## ⌘ Service Archiecture
<table>
  <tr>
    <td><img src="./image/Pipeline.png" alt="Image 1"/></td>
  </tr>
</table>

<br>

## 💿 Data
https://huggingface.co/collections/kfkas/reddit-tip-dataset-681f42af40cff65d89153c88
