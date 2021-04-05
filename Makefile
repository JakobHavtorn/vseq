tests:
	pytest -sv --html=pytest/report.html --self-contained-html --junit-xml=pytest/junit.xml --cov --cov-report=term --cov-report=html:pytest/coverage/html -p no:pytest_wampy tests
