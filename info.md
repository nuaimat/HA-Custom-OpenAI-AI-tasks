## Custom OpenAI AI Tasks Integration

Custom OpenAI AI Tasks is a Home Assistant custom integration that connects the built-in AI Task framework with OpenAI-compatible endpoints. The project began as a fork of the excellent [HA-Azure-AI-Tasks](https://github.com/loryanstrant/HA-Azure-AI-Tasks) integration and has been adapted to focus on OpenAI's public APIs while still supporting Azure OpenAI deployments that expose OpenAI-compatible endpoints.

### Key Features
- Seamless Home Assistant UI setup with config flow support
- Works with OpenAI API keys or Azure OpenAI keys/endpoints that use the OpenAI protocol
- **Image generation** via DALL·E and GPT-Image deployments
- **Attachment-aware** requests for chat/vision models, including camera snapshots and local media
- Options flow for quickly swapping chat and image models without reconfiguring credentials

### Configuration Overview
1. Add this repository as a custom integration in HACS (`https://github.com/nuaimat/HA-Custom-OpenAI-AI-tasks`).
2. Install **Custom OpenAI AI Tasks** and restart Home Assistant.
3. Navigate to **Settings → Devices & Services → Add Integration**.
4. Search for **"Custom OpenAI AI Tasks"**.
5. Provide your endpoint URL (OpenAI or Azure OpenAI) and API key.
6. Enter the deployment/model names for chat and/or image generation.
7. Finish the setup to create the AI Task entity.

### Model Selection
You can configure one or both model types during setup or via the options flow:
- **Chat models**: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-4.1, etc.
- **Image models**: dall-e-2, dall-e-3, gpt-image-1, and compatible custom deployments

### Usage
Once the integration is set up, the AI Task entity can be used in automations and scripts to:
- Generate conversational responses with your chosen chat model
- Create images via DALL·E or GPT-Image with configurable prompts and sizes
- Process multimodal requests that include Home Assistant camera or media attachments

For comprehensive documentation, examples, and release notes, visit the [GitHub repository](https://github.com/nuaimat/HA-Custom-OpenAI-AI-tasks).