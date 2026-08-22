"""데이터 무결성 검증 패키지.

실제 corpus(data/clean/, .gitignore 대상)가 필요해 CI에서는 실행하지 않는다.
문서 재수집·재정제·재색인 후 직접 실행한다.

실행: python -m unittest discover -s data_checks -t . -v
"""
