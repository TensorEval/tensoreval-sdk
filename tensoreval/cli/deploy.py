"""Deploy command — deploy a trained model."""

def deploy_command(run_id: str, name: str, provider: str):
    """Deploy a trained model."""
    from tensoreval.deploy.deployer import Deployer

    print(f"Deploying model '{run_id}' to {provider}...")
    endpoint = Deployer.deploy(
        model_id=run_id,
        name=name or run_id.split("/")[-1],
        provider=provider,
    )

    print(f"\nDeployment complete!")
    print(f"  Model ID: {endpoint.model_id}")
    print(f"  Base URL: {endpoint.base_url}")
    print(f"  Provider: {endpoint.provider}")
    print(f"\nUse with OpenAI client:")
    print(f"  from openai import OpenAI")
    print(f"  client = OpenAI(base_url='{endpoint.base_url}', api_key='...')")
    print(f"  response = client.chat.completions.create(model='{endpoint.model_id}', messages=[...])")
