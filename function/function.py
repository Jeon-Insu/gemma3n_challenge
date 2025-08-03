def run_model_iter(model, datatype, file_path, questions_list, nI=1, question_key=None):
    """
    LLM 모델을 사용하여 screening을 위한 질문 리스트에 대해 반복적으로 답변을 받습니다.
    각 질문 세트 내에서만 조건부로 질문하고, 세트 간에는 독립적으로 실행됩니다.
    
    Args:
        model: LLM 모델 인스턴스
        datatype: 데이터 타입 ("image" 또는 "audio")
        file_path: 파일 경로
        questions_list: 질문 리스트 (해당 세트의 질문들)
        nI: 반복 횟수
        question_key: 질문 세트 키 ("Q1", "Q2", ...)
        
    Returns:
        로그 데이터
    """
    # 질문 리스트 정리
    q_list = [q for q in questions_list if q is not None and q.strip()]
    
    log = []

    if not q_list:
        print(f"Warning: No valid questions provided for {question_key}")
        return log

    for i in range(nI):
        iter_log = {
            "iteration": i + 1,
            "questions": [],
            "question_key": question_key
        }

        # 각 질문 세트 내에서 조건부 질문 실행
        for j, text_q in enumerate(q_list):
            prompt = {
                "role": "user",
                "content": [
                    {"type": datatype, datatype: file_path},
                    {"type": "text", "text": text_q + " Answer only with a number either 9 or 0."},
                ]
            }

            try:
                response = model.send_message(prompt)
                sub_result = int(response)
            except (ValueError, TypeError):
                print(f"Warning: Invalid response '{response}' for {question_key} question {j+1}, defaulting to 0")
                sub_result = 0
            except Exception as e:
                print(f"Warning: Error processing {question_key} question {j+1}: {str(e)}, defaulting to 0")
                sub_result = 0

            iter_log["questions"].append({
                "question_num": j + 1,
                "question_text": text_q,
                "response": sub_result
            })

            # 조건부 질문 로직: 답변이 0이면 현재 세트의 다음 질문을 하지 않음
            # (다른 세트는 독립적으로 실행됨)
            if sub_result == 0:
                print(f"Question set {question_key}: Stopping at question {j+1} due to 0 response")
                break  # 현재 세트의 다음 질문을 하지 않음

        log.append(iter_log)

    return log



