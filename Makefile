MAIN_SHA=$(shell git rev-parse main)

deploy:
	aws deploy create-deployment \
		--application-name BubbleApp \
		--deployment-config-name CodeDeployDefault.OneAtATime \
		--deployment-group-name BubbleApp-DepGrp \
		--description "Commit ${MAIN_SHA}" \
		--github-location repository=CocoaPriest/bubbleai,commitId=${MAIN_SHA}
