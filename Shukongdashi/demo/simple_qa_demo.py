import re
import json
import os
from typing import List, Dict


BASE_DIR = os.path.dirname(__file__)
KB_PATH = os.path.join(BASE_DIR, 'knowledge_base.json')

with open(KB_PATH, 'r', encoding='utf-8') as f:
    KB: Dict[str, Dict[str, List[str]]] = json.load(f)

# Compile regex patterns for the 4 question templates
PATTERNS = [
    # 0. "X 会引起哪些现象？"
    re.compile(r"(?P<subject>.+?)会引起(?:哪些)?现象[?？]"),
    # 1. "X 会遇到什么错误？"
    re.compile(r"(?P<subject>.+?)会遇到(?:什么|哪些)?错误[?？]"),
    # 2. "X 常出现哪些故障？"
    re.compile(r"(?P<subject>.+?)常(?:出现)?(?:哪些)?故障[?？]"),
    # 3. "(ALMXXX) 报警的含义是什么？"
    re.compile(r"(?P<subject>ALM\d{3,})报警的含义是什么[?？]?")
]


def answer(question: str) -> Dict[str, List[str]]:
    """Return answers as a dict with a list under key 'answer'.
    If no answer matches, an empty list is returned.
    """
    for q_type, pattern in enumerate(PATTERNS):
        m = pattern.search(question)
        if m:
            subject = m.group('subject').strip()
            if q_type == 0:  # reason -> phenomena
                return {'answer': KB.get('phenomena_by_reason', {}).get(subject, [])}
            elif q_type == 1:  # operation -> errors
                return {'answer': KB.get('errors_by_operation', {}).get(subject, [])}
            elif q_type == 2:  # part -> faults
                return {'answer': KB.get('faults_by_part', {}).get(subject, [])}
            elif q_type == 3:  # alarm -> meaning
                alarm_code = subject.upper()
                meaning = KB.get('meaning_of_alarm', {}).get(alarm_code, '')
                return {'answer': [meaning] if meaning else []}
    # No template matched
    return {'answer': []}


def main():
    print("简易故障问答 DEMO (输入 'quit' 退出)")
    while True:
        try:
            q = input('\n提问: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n再见!')
            break
        if q.lower() in {"quit", "exit"}:
            print("再见!")
            break
        result = answer(q)
        if result['answer']:
            print("答案:" , "; ".join(result['answer']))
        else:
            print("抱歉, 暂无答案。")


if __name__ == '__main__':
    main()