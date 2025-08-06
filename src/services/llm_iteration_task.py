"""
LLM Iteration Task Service
비디오에서 추출된 이미지와 오디오를 사용하여 LLM 모델로 반복적인 질문을 수행하는 태스크
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import streamlit as st

# Import function module using direct path
import sys
import os
import importlib.util

# Get the path to function.py
function_path = os.path.join(os.path.dirname(__file__), '..', '..', 'function', 'function.py')
spec = importlib.util.spec_from_file_location("function", function_path)
function_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(function_module)

# Import the functions
run_model_iter = function_module.run_model_iter


class LLMIterationTask:
    """LLM Iteration Task class"""
    
    def __init__(self):
        self.task_name = "Screening Task"
        self.task_description = "비디오에서 추출된 이미지와 오디오를 사용하여 screening을 위한 LLM 모델로 반복적인 질문을 수행합니다."
        
    def execute_task(self, model, task_id: str, question_key: str = "Q1", task_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute LLM Iteration Task.
        
        Args:
            model: LLM 모델 인스턴스
            task_id: 태스크 ID
            question_key: 질문 키 ("Q1", "Q2", ...)
            task_params: 태스크 파라미터 (반복 횟수 등)
            
        Returns:
            Dict[str, Any]: 실행 결과
        """
        try:
            # 비디오에서 추출된 파일 경로 가져오기
            video_prefix = st.session_state.get(f'video_prefix_{task_id}')
            if not video_prefix:
                return {"error": "비디오가 녹화되지 않았습니다. 먼저 비디오를 녹화해주세요."}
            
            # 사용자별 폴더 확인 (video_prefix와 동일한 UUID 사용)
            user_id = st.session_state.get('user_id', st.session_state.get('video_prefix', 'default_user'))
            user_record_dir = Path(f"./recordings/{user_id}")
            
            # 폴더가 존재하지 않으면 오류 반환
            if not user_record_dir.exists():
                return {"error": "비디오 녹화 폴더가 존재하지 않습니다. 먼저 비디오를 녹화해주세요."}
            
            # 질문 세트 및 데이터 타입 가져오기
            from config.questions import get_question_set, get_question_data_type
            questions_list = get_question_set(question_key)
            data_type = get_question_data_type(question_key)
            
            if not questions_list:
                return {"error": f"질문 세트 '{question_key}'를 찾을 수 없습니다."}
            
            # 파라미터 설정
            if task_params:
                nI = task_params.get('nI', 1)  # 반복 횟수
            else:
                nI = 1
            
            # 데이터 타입별 파일 경로 설정
            if data_type == "oneframe_image":
                # 중간 프레임 선택 (짝수일 때는 앞쪽, 홀수일 때는 정중앙)
                images_dir = user_record_dir / "images"
                if not images_dir.exists():
                    return {"error": f"이미지 디렉토리를 찾을 수 없습니다: {images_dir}"}
                
                # 모든 프레임 파일 찾기
                frame_files = sorted(list(images_dir.glob(f"{video_prefix}_frame_*.png")))
                if not frame_files:
                    return {"error": f"프레임 파일을 찾을 수 없습니다: {images_dir}"}
                
                # 중간 프레임 선택 로직
                total_frames = len(frame_files)
                if total_frames == 1:
                    # 프레임이 1개일 때: 해당 프레임 선택
                    middle_index = 0
                else:
                    # 프레임이 여러 개일 때: 중간 인덱스 선택 (짝수일 때는 앞쪽)
                    middle_index = (total_frames - 1) // 2
                
                file_path = frame_files[middle_index]
                print(f"🔍 Debug: Selected frame {middle_index + 1} of {total_frames} frames for {question_key}")
                
                if not file_path.exists():
                    return {"error": f"선택된 이미지 파일을 찾을 수 없습니다: {file_path}"}
                
                # LLM 반복 질문 실행 (단일 프레임)
                try:
                    # 단일 프레임 처리 전에 모델 상태 초기화
                    print(f"🔄 Clearing history for {question_key} (oneframe_image)")
                    if hasattr(model, 'clear_history'):
                        try:
                            model.clear_history()
                            print(f"✅ History cleared successfully for {question_key}")
                        except Exception as clear_error:
                            print(f"⚠️ Warning: Failed to clear history for {question_key}: {clear_error}")
                    else:
                        print(f"⚠️ Warning: Model does not have clear_history method for {question_key}")
                    
                    log = run_model_iter(
                        model=model,
                        datatype="image",
                        file_path=str(file_path),
                        questions_list=questions_list,
                        nI=nI,
                        question_key=question_key
                    )
                    
                    # 메모리 정리 및 추가 모델 상태 초기화
                    import gc
                    gc.collect()
                    
                    # 추가 모델 상태 초기화 시도
                    try:
                        if hasattr(model, 'reset'):
                            model.reset()
                            print(f"✅ Model reset completed for {question_key}")
                    except Exception as reset_error:
                        print(f"⚠️ Warning: Model reset failed for {question_key}: {reset_error}")
                    
                    try:
                        if hasattr(model, 'empty_cache'):
                            model.empty_cache()
                            print(f"✅ Model cache cleared for {question_key}")
                    except Exception as cache_error:
                        print(f"⚠️ Warning: Model cache clear failed for {question_key}: {cache_error}")
                    
                except Exception as e:
                    print(f"Warning: Error processing single frame: {str(e)}")
                    # 오류가 발생한 경우 기본 로그 생성
                    log = [{
                        "iteration": 1,
                        "questions": [],
                        "question_key": question_key,
                        "error": str(e)
                    }]
                    
                    # 기본 응답 생성 (모든 질문에 대해 0으로 설정)
                    for j, question_text in enumerate(questions_list):
                        log[0]["questions"].append({
                            "question_num": j + 1,
                            "question_text": question_text,
                            "response": 0
                        })
                
            elif data_type == "allframe_image":
                # 모든 프레임 사용
                images_dir = user_record_dir / "images"
                if not images_dir.exists():
                    return {"error": f"이미지 디렉토리를 찾을 수 없습니다: {images_dir}"}
                
                # 모든 프레임 파일 찾기
                frame_files = sorted(list(images_dir.glob(f"{video_prefix}_frame_*.png")))
                if not frame_files:
                    return {"error": f"프레임 파일을 찾을 수 없습니다: {images_dir}"}
                
                log = []
                for frame_idx, frame_path in enumerate(frame_files):
                    try:
                        # 각 프레임 처리 전에 모델 상태 초기화
                        print(f"🔄 Clearing history for {question_key} (allframe_image, frame {frame_idx})")
                        if hasattr(model, 'clear_history'):
                            try:
                                model.clear_history()
                                print(f"✅ History cleared successfully for {question_key} (frame {frame_idx})")
                            except Exception as clear_error:
                                print(f"⚠️ Warning: Failed to clear history for {question_key} (frame {frame_idx}): {clear_error}")
                        else:
                            print(f"⚠️ Warning: Model does not have clear_history method for {question_key} (frame {frame_idx})")
                        
                        frame_log = run_model_iter(
                            model=model,
                            datatype="image",
                            file_path=str(frame_path),
                            questions_list=questions_list,
                            nI=1,  # 각 프레임당 1회만 실행
                            question_key=question_key
                        )
                        
                        # 프레임 정보 추가
                        for iter_log in frame_log:
                            iter_log["frame_index"] = frame_idx
                            iter_log["frame_path"] = str(frame_path)
                        
                        log.extend(frame_log)
                        
                        # 메모리 정리 및 추가 모델 상태 초기화
                        import gc
                        gc.collect()
                        
                        # 추가 모델 상태 초기화 시도
                        try:
                            if hasattr(model, 'reset'):
                                model.reset()
                                print(f"✅ Model reset completed for {question_key} (frame {frame_idx})")
                        except Exception as reset_error:
                            print(f"⚠️ Warning: Model reset failed for {question_key} (frame {frame_idx}): {reset_error}")
                        
                        try:
                            if hasattr(model, 'empty_cache'):
                                model.empty_cache()
                                print(f"✅ Model cache cleared for {question_key} (frame {frame_idx})")
                        except Exception as cache_error:
                            print(f"⚠️ Warning: Model cache clear failed for {question_key} (frame {frame_idx}): {cache_error}")
                        
                    except Exception as e:
                        print(f"Warning: Error processing frame {frame_idx}: {str(e)}")
                        # 오류가 발생한 프레임에 대한 기본 로그 생성
                        error_log = {
                            "iteration": 1,
                            "questions": [],
                            "question_key": question_key,
                            "frame_index": frame_idx,
                            "frame_path": str(frame_path),
                            "error": str(e)
                        }
                        
                        # 기본 응답 생성 (모든 질문에 대해 0으로 설정)
                        for j, question_text in enumerate(questions_list):
                            error_log["questions"].append({
                                "question_num": j + 1,
                                "question_text": question_text,
                                "response": 0
                            })
                        
                        log.append(error_log)
                
            elif data_type == "audio":
                # 오디오 파일 경로
                file_path = user_record_dir / "audio" / f"{video_prefix}_audio.wav"
                if not file_path.exists():
                    return {"error": f"오디오 파일을 찾을 수 없습니다: {file_path}"}
                
                # LLM 반복 질문 실행 (오디오)
                try:
                    # 오디오 처리 전에 모델 상태 초기화
                    print(f"🔄 Clearing history for {question_key} (audio)")
                    if hasattr(model, 'clear_history'):
                        try:
                            model.clear_history()
                            print(f"✅ History cleared successfully for {question_key}")
                        except Exception as clear_error:
                            print(f"⚠️ Warning: Failed to clear history for {question_key}: {clear_error}")
                    else:
                        print(f"⚠️ Warning: Model does not have clear_history method for {question_key}")
                    
                    log = run_model_iter(
                        model=model,
                        datatype="audio",
                        file_path=str(file_path),
                        questions_list=questions_list,
                        nI=nI,
                        question_key=question_key
                    )
                    
                    # 메모리 정리 및 추가 모델 상태 초기화
                    import gc
                    gc.collect()
                    
                    # 추가 모델 상태 초기화 시도
                    try:
                        if hasattr(model, 'reset'):
                            model.reset()
                            print(f"✅ Model reset completed for {question_key}")
                    except Exception as reset_error:
                        print(f"⚠️ Warning: Model reset failed for {question_key}: {reset_error}")
                    
                    try:
                        if hasattr(model, 'empty_cache'):
                            model.empty_cache()
                            print(f"✅ Model cache cleared for {question_key}")
                    except Exception as cache_error:
                        print(f"⚠️ Warning: Model cache clear failed for {question_key}: {cache_error}")
                    
                except Exception as e:
                    print(f"Warning: Error processing audio: {str(e)}")
                    # 오류가 발생한 경우 기본 로그 생성
                    log = [{
                        "iteration": 1,
                        "questions": [],
                        "question_key": question_key,
                        "error": str(e)
                    }]
                    
                    # 기본 응답 생성 (모든 질문에 대해 0으로 설정)
                    for j, question_text in enumerate(questions_list):
                        log[0]["questions"].append({
                            "question_num": j + 1,
                            "question_text": question_text,
                            "response": 0
                        })
                
            else:
                return {"error": f"지원하지 않는 데이터 타입입니다: {data_type}"}
            
            # 결과 정리
            result = {
                "task_name": self.task_name,
                "task_id": task_id,
                "question_key": question_key,
                "data_type": data_type,
                "file_path": str(file_path) if data_type != "allframe_image" else f"Multiple frames ({len(frame_files) if 'frame_files' in locals() else 0})",
                "questions": questions_list,
                "iterations": nI,
                "log": log,
                "status": "completed"
            }
            
            # ========================================
            # 로그 처리 및 최종 결과 추출 공간
            # ========================================
            # 여기서 log 데이터를 분석하여 최종 결과를 추출합니다.
            # 예: 각 질문별 응답 통계, 전체 점수 계산, 위험도 평가 등
            
            # 로그 분석 로직 실행
            processed_result = self.process_log_data(log, question_key, data_type)
            result["final_result"] = processed_result
            
            # ========================================
            # 로그 처리 공간 끝
            # ========================================
            
            return result
            
        except Exception as e:
            return {
                "error": f"태스크 실행 중 오류가 발생했습니다: {str(e)}",
                "status": "failed"
            }
    
    def get_task_info(self) -> Dict[str, Any]:
        """태스크 정보를 반환합니다."""
        return {
            "name": self.task_name,
            "description": self.task_description,
            "type": "llm_iteration",
            "requires_video": True,
            "requires_model": True
        }
    
    def validate_prerequisites(self, task_id: str) -> Dict[str, Any]:
        """태스크 실행 전 필요한 조건들을 확인합니다."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # 비디오 녹화 확인
        video_prefix = st.session_state.get(f'video_prefix_{task_id}')
        if not video_prefix:
            validation_result["valid"] = False
            validation_result["errors"].append("비디오가 녹화되지 않았습니다.")
        
        # 비디오 분할 완료 확인
        video_split_done = st.session_state.get(f'video_split_done_{task_id}', False)
        if not video_split_done:
            validation_result["valid"] = False
            validation_result["errors"].append("비디오 처리가 완료되지 않았습니다.")
        
        # 모델 초기화 확인
        model_initialized = st.session_state.get('model_initialized', False)
        
        if not model_initialized:
            validation_result["valid"] = False
            validation_result["errors"].append("LLM 모델이 초기화되지 않았습니다. 사이드바에서 모델을 초기화해주세요.")
        
        return validation_result
    
    def process_log_data(self, log: List[Dict], question_key: str, data_type: str) -> Dict[str, Any]:
        """
        로그 데이터를 분석하여 최종 결과를 추출합니다.
        
        Args:
            log: LLM 실행 로그 데이터
            question_key: 질문 키
            data_type: 데이터 타입
            
        Returns:
            Dict[str, Any]: 처리된 최종 결과
        """
        # ========================================
        # 로그 분석 및 최종 결과 추출 로직
        # ========================================
        
        # 기본 통계 계산
        total_iterations = len(log)
        total_questions = 0
        total_responses = 0
        response_counts = {0: 0, 1: 0}  # 0, 1, 응답 카운트
        
        # 각 반복별 분석
        for iteration in log:
            questions = iteration.get('questions', [])
            total_questions += len(questions)
            
            for question in questions:
                response = question.get('response', 0)
                total_responses += 1
                response_counts[response] = response_counts.get(response, 0) + 1
        
        # 응답 비율 계산
        response_rates = {}
        if total_responses > 0:
            for response, count in response_counts.items():
                response_rates[response] = count / total_responses
        
        # 데이터 타입별 추가 분석
        if data_type == "allframe_image":
            # 프레임별 분석
            frame_analysis = {}
            for iteration in log:
                frame_index = iteration.get('frame_index', 0)
                if frame_index not in frame_analysis:
                    frame_analysis[frame_index] = []
                frame_analysis[frame_index].extend([q.get('response', 0) for q in iteration.get('questions', [])])
        
        # ========================================
        # Feature 점수 계산 로직
        # ========================================
        
        # Feature 변수 초기화
        image_feature = 0
        audio_feature = 0
        text_feature = ""
        Q15_result = None
        
        # 개별 질문별 결과 변수 초기화
        Q1_result = 0
        Q2_result = 0
        Q3_result = 0
        Q4_result = 0
        Q5_result = 0
        Q6_result = 0
        Q7_result = 0
        Q8_result = 0
        Q9_result = 0
        Q10_result = 0
        Q11_result = 0
        Q12_result = 0
        Q13_result = 0
        
        # 질문별 응답 처리
        if question_key == "Q1":
            # Q1: 결과값을 image_feature에 더하기
            for iteration in log:
                for question in iteration.get('questions', []):
                    response = question.get('response', 0)
                    if response == 9:
                        
                        Q1_result = 1
        elif question_key == "Q2":
            # Q2: Q2와 Q2_1이 모두 9인 경우에만 -1
            questions_responses = []
            for iteration in log:
                for question in iteration.get('questions', []):
                    questions_responses.append(question.get('response', 0))
            
            if len(questions_responses) >= 2:
                if questions_responses[0] == 9 and questions_responses[1] == 9:
                    Q2_result = -1
                else:
                    Q2_result = 0
                    
        elif question_key == "Q3":
            # Q3: 각 프레임별로 Q3와 Q3_1이 모두 9인 경우를 확인
            frame_scores = []
            for iteration in log:
                frame_index = iteration.get('frame_index', 0)
                questions = iteration.get('questions', [])
                
                # 두 번째 질문까지 있는 경우만 카운트 (첫 번째 질문이 9인 경우)
                if len(questions) >= 2:
                    if questions[0].get('response', 0) == 9 and questions[1].get('response', 0) == 9:
                        frame_scores.append(1)
                    else:
                        frame_scores.append(0)
            
            if frame_scores:
                avg_score = sum(frame_scores) / len(frame_scores)
                if avg_score > 0.15:  # 70% 이상의 프레임에서 조건 만족
                    
                    Q3_result = 1
                    
        elif question_key == "Q4":
            # Q4: 모든 프레임의 평균이 0.3보다 작으면 1 더하기
            total_response = 0
            frame_count = 0
            for iteration in log:
                for question in iteration.get('questions', []):
                    total_response += question.get('response', 0)
                frame_count += 1
            
            if frame_count > 0:
                avg_response = total_response / frame_count
                if avg_response < 0.15:
                    
                    Q4_result = 1
                    
        elif question_key == "Q5":
            # Q5: 모든 프레임의 평균이 0.7보다 크면 1 더하기
            total_response = 0
            frame_count = 0
            for iteration in log:
                for question in iteration.get('questions', []):
                    total_response += question.get('response', 0)
                frame_count += 1
            
            if frame_count > 0:
                avg_response = total_response / frame_count
                if avg_response > 0.7:
                    
                    Q5_result = 1
                    
        elif question_key == "Q6":
            # Q6: 결과값을 image_feature에 더하기
            for iteration in log:
                for question in iteration.get('questions', []):
                    response = question.get('response', 0)
                    
                    if response == 9:
                        Q6_result = 1
                    
        elif question_key == "Q7":
            # Q7: 모든 프레임의 평균이 0.7보다 크면 1 더하기
            total_response = 0
            frame_count = 0
            for iteration in log:
                for question in iteration.get('questions', []):
                    total_response += question.get('response', 0)
                frame_count += 1
            
            if frame_count > 0:
                avg_response = total_response / frame_count
                if avg_response > 0.15:
                    
                    Q7_result = 1
                    
        elif question_key == "Q8":
            # Q8: 결과값을 Q8_result에 저장
            for iteration in log:
                for question in iteration.get('questions', []):
                    response = question.get('response', 0)
                    if response == 9:
                        Q8_result = 1
                    
        elif question_key == "Q9":
            # Q9: 결과값을 Q9_result에 저장
            for iteration in log:
                for question in iteration.get('questions', []):
                    response = question.get('response', 0)
                    if response == 9:
                        Q9_result = 1
                    
        elif question_key == "Q10":
            # Q10: 결과값을 Q10_result에 저장 (9인 경우 -1로 처리)
            for iteration in log:
                for question in iteration.get('questions', []):
                    response = question.get('response', 0)
                    if response == 9:
                        Q10_result = -1
                    else:
                        Q10_result = 0
                    
        elif question_key == "Q11":
            # Q11: 결과값을 Q11_result에 저장
            for iteration in log:
                for question in iteration.get('questions', []):
                    response = question.get('response', 0)
                    if response == 9:
                        Q11_result = 1
                    
        elif question_key == "Q12":
            # Q12: 결과값을 Q12_result에 저장
            for iteration in log:
                for question in iteration.get('questions', []):
                    response = question.get('response', 0)
                    if response == 9:
                        Q12_result = 1
                    
        elif question_key == "Q13":
            # Q13: 결과값을 Q13_result에 저장
            for iteration in log:
                for question in iteration.get('questions', []):
                    response = question.get('response', 0)
                    if response == 9:
                        Q13_result = 1
                    
        elif question_key == "Q14":
            # Q14: text_feature 설정
            # 각 반복에서 Q14의 실제 실행된 질문 응답을 수집
            q14_responses = []
            q14_1_responses = []
            q14_2_responses = []
            
            for iteration in log:
                questions = iteration.get('questions', [])
                
                # 첫 번째 질문은 항상 있음
                if len(questions) >= 1:
                    q14_responses.append(questions[0].get('response', 0))  # Q14
                
                # 두 번째 질문이 있는 경우
                if len(questions) >= 2:
                    q14_1_responses.append(questions[1].get('response', 0))  # Q14_1
                
                # 세 번째 질문이 있는 경우
                if len(questions) >= 3:
                    q14_2_responses.append(questions[2].get('response', 0))  # Q14_2
            
            # 가장 빈번한 응답을 사용
            from collections import Counter
            
            q14_final = Counter(q14_responses).most_common(1)[0][0] if q14_responses else 0
            q14_1_final = Counter(q14_1_responses).most_common(1)[0][0] if q14_1_responses else 0
            q14_2_final = Counter(q14_2_responses).most_common(1)[0][0] if q14_2_responses else 0
            
            # text_feature 결정 로직
            if q14_final == 0:
                text_feature = "identification"
            else:
                if q14_1_final == 0:
                    text_feature = "eccentricity"
                else:
                    if q14_2_final == 9:
                        text_feature = "optimism"
                    else:
                        text_feature = "none"
              
        elif question_key == "Q15":
            # Q15: Q15_result에 결과값 저장
            for iteration in log:
                for question in iteration.get('questions', []):
                    Q15_result = question.get('response', 0)
        
        # 최종 결과 구성 (feature 점수는 execute_screening_task에서 계산)
        final_result = {
            "question_key": question_key,
            "data_type": data_type,
            "total_iterations": total_iterations,
            "total_questions": total_questions,
            "total_responses": total_responses,
            "response_counts": response_counts,
            "response_rates": response_rates,
            "analysis_summary": {
                "zero_rate": response_rates.get(0, 0),
                "one_rate": response_rates.get(1, 0),
                "nine_rate": response_rates.get(9, 0)
            },
            "question_results": {
                "Q1_result": Q1_result,
                "Q2_result": Q2_result,
                "Q3_result": Q3_result,
                "Q4_result": Q4_result,
                "Q5_result": Q5_result,
                "Q6_result": Q6_result,
                "Q7_result": Q7_result,
                "Q8_result": Q8_result,
                "Q9_result": Q9_result,
                "Q10_result": Q10_result,
                "Q11_result": Q11_result,
                "Q12_result": Q12_result,
                "Q13_result": Q13_result
            },
            "text_feature": text_feature,
            "Q15_result": Q15_result
        }
        
        # 데이터 타입별 추가 정보
        if data_type == "allframe_image":
            final_result["frame_analysis"] = frame_analysis
        
        # ========================================
        # 로그 분석 로직 끝
        # ========================================
        
        return final_result
    
    def calculate_screening_scores(self, all_results: Dict[str, Dict]) -> Dict[str, int]:
        """
        모든 질문 세트의 결과를 종합하여 screening 문항 9개의 점수를 계산합니다.
        
        Args:
            all_results: 모든 질문 세트의 결과 딕셔너리 {question_key: result}
            
        Returns:
            Dict[str, int]: screening 문항 9개의 점수
        """
        # 이미 누적된 값이 있는 결과에서 직접 가져오기
        cumulative_result = None
        for result in all_results.values():
            if result.get('status') == 'completed' and 'final_result' in result:
                cumulative_result = result
                break
        
        if not cumulative_result:
            return {}
        
        # 이미 누적된 값들 직접 사용
        final_result = cumulative_result['final_result']
        cumulative_question_results = final_result.get('question_results', {})
        cumulative_feature_scores = final_result.get('feature_scores', {})
        
        # Image_score와 Audio_score 계산 (누적된 값 사용)
        image_score = 1 if cumulative_feature_scores.get('image_feature', 0) >= 2 else 0
        audio_score = 1 if cumulative_feature_scores.get('audio_feature', 0) >= 3 else 0
        
        # Screening 문항별 점수 계산 (누적된 값 사용)
        screening_scores = {}
        
        # 문항 1: 흥미저하 (Q1 또는 Q3 둘 중 하나라도 1이면 1)
        q1_result = cumulative_question_results.get('Q1_result', 0)
        q3_result = cumulative_question_results.get('Q3_result', 0)
        screening_scores['interest_loss'] = 1 if (q1_result == 1 or q3_result == 1) else 0

        # 문항 2: 우울문항 (image_score 또는 audio_score 둘 중 하나라도 1이면 1)
        screening_scores['depression'] = 1 if (image_score == 1 or audio_score == 1) else 0

        # 문항 3: 잠 문항 (0으로 고정)
        screening_scores['sleep'] = 0

        # 문항 4: 피로 문항 (Q3 또는 Q4 또는 Q8 또는 Q9 중 1개라도 1이면 1)
        q3_fatigue = cumulative_question_results.get('Q3_result', 0)
        q4_fatigue = cumulative_question_results.get('Q4_result', 0)
        q8_fatigue = cumulative_question_results.get('Q8_result', 0)
        q9_fatigue = cumulative_question_results.get('Q9_result', 0)
        screening_scores['fatigue'] = 1 if (q3_fatigue == 1 or q4_fatigue == 1 or q8_fatigue == 1 or q9_fatigue == 1) else 0

        # 문항 5: 입맛 (Q6이 1이면 1)
        q6_appetite = cumulative_question_results.get('Q6_result', 0)
        screening_scores['appetite'] = 1 if q6_appetite == 1 else 0

        # 문항 6: 부정적관련 (Text가 optimism이면 0, 아니면 Q1 또는 Q7 중 하나라도 1이면 1)
        if cumulative_feature_scores.get('text_feature') == "optimism":
            screening_scores['negative_thoughts'] = 0
        else:
            q1_negative = cumulative_question_results.get('Q1_result', 0)
            q7_negative = cumulative_question_results.get('Q7_result', 0)
            screening_scores['negative_thoughts'] = 1 if (q1_negative == 1 or q7_negative == 1) else 0

        # 문항 7: 집중력 저하 (Q12 또는 Q13 중 하나라도 1이면 1)
        q12_concentration = cumulative_question_results.get('Q12_result', 0)
        q13_concentration = cumulative_question_results.get('Q13_result', 0)
        screening_scores['concentration'] = 1 if (q12_concentration == 1 or q13_concentration == 1) else 0

        # 문항 8: 느려짐 (Q9의 값이 1이면 1)
        q9_slowness = cumulative_question_results.get('Q9_result', 0)
        screening_scores['slowness'] = 1 if q9_slowness == 1 else 0

        # 문항 9: 자살사고 (Q15_result가 9이면 1)
        screening_scores['suicidal_thoughts'] = 1 if cumulative_feature_scores.get('Q15_result') == 9 else 0
        
        return screening_scores
    
    def calculate_diagnosis(self, all_results: Dict[str, Dict]) -> str:
        """
        모든 결과를 종합하여 최종 진단을 계산합니다.
        
        Args:
            all_results: 모든 질문 세트의 결과 딕셔너리 {question_key: result}
            
        Returns:
            str: "depressive" 또는 "normal"
        """
        # 이미 누적된 값이 있는 결과에서 직접 가져오기
        cumulative_result = None
        for result in all_results.values():
            if result.get('status') == 'completed' and 'final_result' in result:
                cumulative_result = result
                break
        
        if not cumulative_result:
            return "Typical Range"
        
        # 이미 누적된 값들 직접 사용
        final_result = cumulative_result['final_result']
        cumulative_feature_scores = final_result.get('feature_scores', {})
        
        # Image_score와 Audio_score 계산 (누적된 값 사용)
        image_score = 1 if cumulative_feature_scores.get('image_feature', 0) >= 2 else 0
        audio_score = 1 if cumulative_feature_scores.get('audio_feature', 0) >= 3 else 0
        
        # Screening 점수 계산 (누적된 값 사용)
        screening_scores = self.calculate_screening_scores(all_results)
        screening_total = sum(screening_scores.values())
        
        # 진단 로직 (우선순위 순서대로)
        # 1. text_feature == 'identification' OR Q15_result == 9 → DEPRESSIVE
        if (cumulative_feature_scores.get('text_feature') == "identification" or 
            cumulative_feature_scores.get('Q15_result') == 9):
            return "DEPRESSIVE"
        
        # 2. text_feature == 'optimism' → NORMAL(OPTIMISM)
        elif cumulative_feature_scores.get('text_feature') == "optimism":
            return "NORMAL(OPTIMISM)"
        
        # 3. image_score == 1 AND audio_score == 1 → DEPRESSIVE
        elif image_score == 1 and audio_score == 1:
            return "DEPRESSIVE"
        
        # 4. image_score == 0 AND audio_score == 0 → NORMAL
        elif image_score == 0 and audio_score == 0:
            return "NORMAL"
        
        # 5. screening_total ≥ 7 → CLINICAL EVALUATION ADVISED
        elif screening_total >= 7:
            return "CLINICAL EVALUATION ADVISED"
        
        # 6. screening_total ≥ 4 → MONITORING SUGGESTED
        elif screening_total >= 4:
            return "MONITORING SUGGESTED"
        
        # 7. Otherwise → TYPICAL RANGE
        else:
            return "TYPICAL RANGE" 