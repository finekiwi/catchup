## Tier별 설계 근거

### Tier 1 (단일 사실 조회)
| Q# | 질문 요약 | 소스 파일 | 해당 셀 | 선정 이유 |
| --- | --- | --- | --- | --- |
| Q1 | Knowledge Graph의 정의와 장점 | `LLM_034_neo4j_intro.ipynb` | markdown cell `5` | 그래프 구조, 노드/관계, 맥락 보존이라는 핵심 정의가 한 셀에 명시되어 있어 단일 사실 조회 문항으로 적합하다. |
| Q2 | ToolMessage의 역할과 핵심 필드 | `LLM_016_ToolCalling_Agent.ipynb` | markdown cell `30` | `content`, `name`, `tool_call_id`가 표로 정리되어 있어 에이전트 개념 설명을 가장 안정적으로 회수할 수 있다. |
| Q3 | RAG의 runtime 구성요소 4개 | `day53_langchain.ipynb` | markdown cell `5` | 사전 준비 단계와 runtime 단계가 표로 나뉘어 있고 Retriever, Prompt, LLM, Chain이 직접 나열되어 있다. |
| Q4 | Nori의 의미와 `손해보험회사` 토큰화 결과 | `elk_01_basic.ipynb` | markdown cell `21` | 정의와 예시 토큰 분해 결과가 함께 들어 있어 한국어 형태소 분석기 개념을 단일 셀에서 바로 검증할 수 있다. |

### Tier 2 (코드 output/수치)
| Q# | 질문 요약 | 소스 파일 | 해당 셀 | Before가 실패하는 이유 |
| --- | --- | --- | --- | --- |
| Q5 | `m_train`, `m_test`, `num_px`, 이미지 크기 | `Logistic_Regression_with_a_Neural_Network_mindset.ipynb` | code cell `10` | 출력이 여러 줄 프린트 문자열로 이어져 있어 flat text 추출 시 변수명과 숫자의 대응이 흐려지기 쉽다. |
| Q6 | `"hello world!"` vs `"hello world"` 임베딩 cosine similarity와 문서 임베딩 shape | `day53_langchain.ipynb` | code cell `9`, `10` | 연속된 두 output 셀을 함께 읽어야 하며, 하나는 스칼라 수치이고 하나는 shape이라 baseline chunking에서 결합이 자주 깨진다. |
| Q7 | `insurance` 인덱스 bulk 건수와 `RatDcd="C"` 검색 hit 수 | `elk_01_basic.ipynb` | code cell `16`, `19` | bulk 완료 로그와 검색 결과 수가 떨어진 셀에 있어 output-질문 정합성이 약해지고, 표 출력까지 섞여 값 회수가 불안정하다. |
| Q8 | 모든 Topic 노드 조회 결과의 Topic 이름 3개 | `LLM_034_neo4j_intro.ipynb` | code cell `29` | Python 리스트/딕셔너리 형태의 execute result를 정확히 파싱해야 하므로 단순 텍스트 추출에서는 값 경계가 쉽게 무너진다. |

### Tier 3 (교차 문서)
| Q# | 질문 요약 | 소스 파일 조합 | 연결 개념 | 자연스러운 연결인지 |
| --- | --- | --- | --- | --- |
| Q9 | `search_db` 도구가 day53의 RAG 파이프라인을 어떻게 구현하는가 | `LLM_016_ToolCalling_Agent.ipynb` + `day53_langchain.ipynb` | retriever를 tool로 감싼 agent형 RAG | 자연스럽다. day53이 Retriever/RAG 구조를 설명하고, LLM_016이 그 retrieval step을 `search_db` tool로 실제 에이전트 루프에 삽입한다. |
| Q10 | 삼성전자 기술 질문에서 GraphCypherQAChain이 벡터 검색보다 적합한 이유 | `LLM_034_neo4j_intro.ipynb` + `day53_langchain.ipynb` | graph relation query vs semantic similarity search | 자연스럽다. 하나는 명시적 관계 질의, 다른 하나는 임베딩 기반 유사도 검색이라 문제 유형에 따른 retrieval 선택을 비교하기 좋다. |
| Q11 | Elasticsearch와 LangChain vector retrieval이 어떻게 상호보완되는가 | `elk_01_basic.ipynb` + `day53_langchain.ipynb` | exact field filtering + semantic retrieval | 자연스럽다. `term/range/bool`과 dense retrieval/MMR은 실제 RAG 시스템에서 자주 결합되는 두 축이다. |
| Q12 | 하나의 assistant에서 tool agent, graph backend, structured search backend를 어떻게 조합하는가 | `LLM_016_ToolCalling_Agent.ipynb` + `LLM_034_neo4j_intro.ipynb` + `elk_01_basic.ipynb` | orchestration + multi-backend retrieval | 자연스럽다. LLM_016은 orchestration을, Neo4j와 Elasticsearch 노트북은 서로 다른 tool backend를 제공하므로 실제 시스템 설계 관점 연결이 성립한다. |

### Tier 4 (코드셀 구조 의존)
| Q# | 질문 요약 | 소스 파일 | 해당 코드셀 | nbformat 없이 답 불가능한 이유 |
| --- | --- | --- | --- | --- |
| Q13 | `model()` 내부에서 최종 예측이 만들어지는 함수 호출 순서 | `Logistic_Regression_with_a_Neural_Network_mindset.ipynb` | code cell `20`, `27`, `31`, `34`, `38` | `model -> optimize -> propagate -> sigmoid`와 `predict` 호출 순서를 여러 함수 정의 셀에 걸쳐 추적해야 해서, 셀 구조를 보지 않으면 정확한 call graph를 복원하기 어렵다. |
| Q14 | `nlp_keywords.txt` 기반 chain에서 `reference`가 어떻게 구성되는가 | `day53_langchain.ipynb` | code cell `78` | `itemgetter`, `retriever_nlp`, `RunnableLambda(reorder_documents)`, `PromptTemplate`, `ChatOpenAI`, `StrOutputParser`의 runnable composition을 코드 수준에서 읽어야만 답할 수 있다. |
| Q15 | `create_agent` 날씨 예제에서 반환된 메시지 상태의 순서 | `LLM_016_ToolCalling_Agent.ipynb` | code cell `44`, `45` | `HumanMessage -> AIMessage(tool_call) -> ToolMessage -> AIMessage`라는 typed state ordering은 실행 결과 객체 구조를 봐야 하며, 평면 텍스트만으로는 메시지 타입과 순서를 안정적으로 식별하기 어렵다. |
