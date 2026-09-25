import os

import aws_cdk as cdk

from stack import RagAssistantStack

app = cdk.App()
RagAssistantStack(
    app,
    "OrbitusRagAssistant",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-2"),
    ),
)
app.synth()
