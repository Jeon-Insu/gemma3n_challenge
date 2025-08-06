"""
질문 설정 파일
Define questions used in Screening Tasks.
"""

# 개별 질문 정의
QUESTIONS = {
    "Q1" : "If the appearance is unkempt, with messy clothing or hair and signs of self-neglect, assign as 9; otherwise, assign as 0.Tell me reason. Answer the only 9 or 0.",
    "Q2" : "If the person is a male, assign 9; otherwise, assign 0. Tell me reason. Answer the only 9 or 0.",
    "Q2_1" : "If the person in the image is in red, pink, yellow or green, answer 9; otherwise, answer 0. Tell me reason. Answer the only 9 or 0.",
    "Q3" : "If the person has their eyes open, assign 9; otherwise, assign 0. Tell me reason. Answer the only 9 or 0.",
    "Q3_1" : "If the person has dull eyes, assign 9; otherwise, assign 0. Tell me reason. Answer the only 9 or 0.",
    "Q4" : "If clear teeth are visible in the image, assign 9; otherwise, assign 0. Tell me reason. Answer the only 9 or 0.",
    "Q5" : "If the person is in a straight, upright posture, assign 0; otherwise, assign 9. Tell me reason. Answer the only 9 or 0.",
    "Q6" : "If the person's face and upper body appear very thin or very overweight, assign 9; otherwise, assign 0. Tell me reason. Answer the only 9 or 0. Tell me reason. Answer the only 9 or 0.",
    "Q7" : "If the person's posture appears defensive — for example, crossed arms, hunched shoulders, furrowed brow, or other signs of physical withdrawal or tension — assign 9; otherwise, assign 0.",
    "Q8" : "If the person's voice is weak, assign 9; otherwise, assign 0. Tell me reason. Answer the only 9 or 0.",
    "Q9" : "If the person's speech is slow, assign 9; otherwise, assign 0. Tell me reason. Answer the only 9 or 0.",
    "Q10" : "If the person's speech is very fast, assign 9; otherwise, assign 0. Tell me reason. Answer the only 9 or 0.",
    "Q11" : "If the person's voice sounds dull, assign 9; otherwise, assign 0. Tell me reason. Answer the only 9 or 0.",
    "Q12" : "If the speaker speaks in a sentence consisting of not more than 3 words more than 2 times, assign 9; otherwise, assign 0. Tell me reason. Answer the only 9 or 0.",
    "Q13" : "If the person makes silence of over 2 seconds between voice more than 3 times, assign 9; otherwise, assign 0. Tell me reason. Answer the only 9 or 0.",
    "Q14" : "If the speaker, while describing a picture of a house on fire, displays extreme expressions, output 0. Otherwise, if the description and contents seems appropriate, output 9. Tell me reason. Answer the only 9 or 0.",
    "Q14_1" : "If the speaker, while describing the picture of a house on fire, expresses general concerns about the fire or victims, assign 9; otherwise assign 0. Tell me reason. Answer the only 9 or 0.",
    "Q14_2" : "If the speaker, while describing a picture of a house on fire, expresses a detailed positive interpretation, output 9. Otherwise, output 0. Tell me reason. Answer the only 9 or 0.",
    "Q15" : "If the speaker, while describing a picture of a house on fire, evoke suicidal thoughts(e.g. 'I want to die'), assign 9; otherwise, assign 0. Tell me reason. Answer the only 9 or 0.",
}

# 질문 분류
ONEFRAME_IMAGE_QUESTIONS = {"Q1","Q2","Q6"}
ALLFRAME_IMAGE_QUESTIONS = {"Q3","Q4","Q5","Q7"}
AUDIO_QUESTIONS = {"Q8","Q9","Q10","Q11","Q12","Q13","Q14","Q15"}

# 15개 질문 세트 정의 (서브 항목 포함)
QUESTION_SETS = {
    "Q1": [QUESTIONS["Q1"]],
    "Q2": [QUESTIONS["Q2"], QUESTIONS["Q2_1"]],
    "Q3": [QUESTIONS["Q3"], QUESTIONS["Q3_1"]],
    "Q4": [QUESTIONS["Q4"]],
    "Q5": [QUESTIONS["Q5"]],
    "Q6": [QUESTIONS["Q6"]],
    "Q7": [QUESTIONS["Q7"]],
    "Q8": [QUESTIONS["Q8"]],
    "Q9": [QUESTIONS["Q9"]],
    "Q10": [QUESTIONS["Q10"]],
    "Q11": [QUESTIONS["Q11"]],
    "Q12": [QUESTIONS["Q12"]],
    "Q13": [QUESTIONS["Q13"]],
    "Q14": [QUESTIONS["Q14"], QUESTIONS["Q14_1"], QUESTIONS["Q14_2"]],
    "Q15": [QUESTIONS["Q15"]]
}

# 질문 세트별 데이터 타입 매핑
QUESTION_DATA_TYPES = {
    "Q1": "oneframe_image",
    "Q2": "oneframe_image", 
    "Q3": "allframe_image",
    "Q4": "allframe_image",
    "Q5": "allframe_image",
    "Q6": "oneframe_image",
    "Q7": "allframe_image",
    "Q8": "audio",
    "Q9": "audio",
    "Q10": "audio",
    "Q11": "audio",
    "Q12": "audio",
    "Q13": "audio",
    "Q14": "audio",
    "Q15": "audio"
}

def get_question_set(question_key: str) -> list:
    """
    지정된 질문 세트를 반환합니다.
    
    Args:
        question_key: 질문 키 ("Q1", "Q2", ...)
        
    Returns:
        질문 리스트
    """
    return QUESTION_SETS.get(question_key, [])

def get_question_data_type(question_key: str) -> str:
    """
    질문 세트의 데이터 타입을 반환합니다.
    
    Args:
        question_key: 질문 키
        
    Returns:
        데이터 타입 ("oneframe_image", "allframe_image", "audio")
    """
    return QUESTION_DATA_TYPES.get(question_key, "oneframe_image")

def get_all_question_keys() -> list:
    """
    모든 질문 키를 반환합니다.
    
    Returns:
        질문 키 리스트
    """
    return list(QUESTION_SETS.keys())

def get_question_count(question_key: str) -> int:
    """
    질문 세트의 질문 개수를 반환합니다.
    
    Args:
        question_key: 질문 키
        
    Returns:
        질문 개수
    """
    return len(get_question_set(question_key))
