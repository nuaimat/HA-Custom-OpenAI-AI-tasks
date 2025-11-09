# Custom OpenAI AI Tasks - Home Assistant Integration

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/custom-components/hacs)
[![GitHub release](https://img.shields.io/github/release/nuaimat/HA-Custom-OpenAI-AI-tasks.svg)](https://github.com/nuaimat/HA-Custom-OpenAI-AI-tasks/releases)

Custom OpenAI AI Tasks is a Home Assistant custom integration that connects the built-in AI Task framework with OpenAI-compatible endpoints. It started life as a fork of [loryanstrant/HA-Azure-AI-Tasks](https://github.com/loryanstrant/HA-Azure-AI-Tasks) and has been reworked to focus on OpenAI's public APIs while remaining compatible with Azure OpenAI deployments that expose the OpenAI protocol.

> ✨ **Why this fork?** The Home Assistant AI Task ecosystem is rapidly expanding. This fork keeps the familiar experience from the Azure integration but lowers the barrier for users who rely on OpenAI API keys or custom OpenAI-compatible deployments.

<p align="center"><img width="256" height="256" alt="icon" src="https://github.com/user-attachments/assets/934b88ed-f038-474f-9211-4417717e5e84" /><p>



## Features

- Guided configuration through the Home Assistant UI
- Secure API key storage handled by Home Assistant
- **Flexible chat models** – GPT-4o, GPT-4.1, GPT-4o-mini, GPT-4 Turbo, and any compatible deployment name
- **🎨 Image generation** – DALL·E 2, DALL·E 3, GPT-Image-1, or compatible custom deployments
- **Attachment-aware interactions** – process camera snapshots, uploaded media, and URLs alongside prompts
- **Separate chat & image slots** – configure chat-only, image-only, or combined entities
- **Options flow for quick model swaps** – change deployments without re-entering credentials
- **Multiple entries supported** – connect to different endpoints/keys for specialized tasks
- HACS ready for easy installation

## Background

This repository retains the polished UX from the original Azure integration while focusing on:

1. Native OpenAI API usage (api.openai.com)
2. Azure OpenAI deployments that expose the official OpenAI protocol
3. Community-hosted OpenAI-compatible backends (e.g., proxy services)

The codebase has been cleaned up, renamed, and aligned with the new domain `custom_openai_ai_tasks` to better reflect its purpose inside Home Assistant.

## Installation

### Via HACS (Recommended)

1. Open HACS in your Home Assistant instance
2. Go to "Integrations"
3. Click the three dots menu and select "Custom repositories"
4. Add `https://github.com/nuaimat/HA-Custom-OpenAI-AI-tasks` as the repository
5. Set category to "Integration"
6. Click "Add"
7. Find **"Custom OpenAI AI Tasks"** in the integration list and install it
8. Restart Home Assistant
9. Go to Configuration > Integrations
10. Click "+ Add Integration" and search for "Custom OpenAI AI Tasks"
11. Press Submit to complete the installation.

Or replace steps 1-6 with this:

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=nuaimat&repository=HA-Custom-OpenAI-AI-tasks&category=integration)

### Manual Installation

1. Copy the `custom_components/custom_openai_ai_tasks` folder to your Home Assistant `custom_components` directory
2. Restart Home Assistant
3. Add the integration through the UI (Settings → Devices & Services → Add Integration)

## Configuration

1. Go to **Settings → Devices & Services → Add Integration**
2. Search for **"Custom OpenAI AI Tasks"**
3. Provide your endpoint URL:
   - For OpenAI: `https://api.openai.com/v1`
   - For Azure OpenAI: `https://YOUR-RESOURCE-NAME.openai.azure.com`
   - For self-hosted/proxy endpoints: the base URL that mimics OpenAI's API
4. Enter your API key (OpenAI key, Azure key, or custom proxy key)
5. **Enter your preferred chat model** (e.g., `gpt-4o`, `gpt-4.1`, `gpt-4o-mini`). Leave blank to disable chat.
6. **Enter your preferred image model** (e.g., `dall-e-3`, `gpt-image-1`). Leave blank to disable image generation.
7. Give your integration a name (defaults to **Custom OpenAI AI Tasks**)
8. Click **Submit**

<img width="383" height="478" alt="image" src="https://github.com/user-attachments/assets/8932c51a-8fcb-42bc-9e22-ead143c610d7" />






> 💡 **Azure OpenAI users:** enter the deployment names you created in Azure, not the base model names. The integration forwards requests using the OpenAI-style `/deployments/{deployment}/...` routes.

![Config flow screenshot showing Custom OpenAI AI Tasks setup](https://github.com/user-attachments/assets/8932c51a-8fcb-42bc-9e22-ead143c610d7)


### Reconfiguration

To change AI models without re-entering credentials:
1. Go to your **Custom OpenAI AI Tasks** integration
2. Click "Configure" 
3. Enter different chat/image models as needed (use placeholder text `[None - leave empty to disable chat]` or `[None - leave empty to disable images]` to clear fields)
4. Save changes

**Tip**: You can create specialized entities by leaving one model type empty:
- **Chat-only entities**: configure chat model, leave image model empty
- **Image-only entities**: configure image model, leave chat model empty  
- **Combined entities**: configure both models

<img width="1072" height="700" alt="image" src="https://github.com/user-attachments/assets/598b8c28-7663-4507-be63-22413cac4b9d" />



## Usage

Once configured, the integration provides an AI Task entity that can be used in automations and scripts to process AI tasks using your OpenAI-compatible endpoint.



### Chat/Text Generation
Example service call for generating text responses:
```yaml
service: ai_task.process
target:
  entity_id: ai_task.custom_openai_ai_tasks
data:
  task: "Summarize the weather forecast for today"
```
![HA Azure AI Task example](https://github.com/user-attachments/assets/592ec039-20ea-436f-a6f0-caf88bef9b56)

### 🎨 Image Generation
Example service calls for generating images with DALL-E:

**Basic Image Generation:**
```yaml
action: ai_image.generate_image
data:
  prompt: "A futuristic smart home with holographic displays and AI assistants"
  entity_id: ai_image.custom_openai_ai_tasks_dall_e_3
```
<img width="1413" height="1164" alt="image" src="https://github.com/user-attachments/assets/b1d7c898-bb54-4398-838f-838f4a5e26fa" />
<img width="1360" height="1246" alt="Generated AI image of a futuristic smart home with holographic displays and AI assistants" src="https://github.com/user-attachments/assets/f748fffd-a379-4745-845f-1fecdff31e44" />


**Advanced Image Generation with Parameters:**
```yaml
action: ai_image.generate_image
data:
  prompt: "A cozy living room during sunset with warm lighting"
  entity_id: ai_image.custom_openai_ai_tasks_dall_e_3
  size: "1024x1024"
  quality: "hd"
  style: "vivid"
```

**Supported DALL-E Parameters:**
- **size**: Image dimensions (DALL-E 2: 256x256, 512x512, 1024x1024; DALL-E 3: 1024x1024, 1024x1792, 1792x1024)
- **quality**: Image quality for DALL-E 3 (standard, hd)  
- **style**: Image style for DALL-E 3 (natural, vivid)
- **n**: Number of images to generate (1-10 for DALL-E 2, 1 for DALL-E 3)



### Image/Video Analysis with Attachments
Example service calls for analyzing images or camera streams:

**Analyze Camera Stream:**
```yaml
action: ai_task.generate_data
data:
  task_name: camera analysis
  instructions: What's going on in this picture?
  entity_id: ai_task.custom_openai_ai_tasks
  attachments:
    media_content_id: media-source://camera/camera.front_door_fluent
    media_content_type: application/vnd.apple.mpegurl
    metadata:
      title: Front door camera
      media_class: video
```
<img width="1390" height="1247" alt="image" src="https://github.com/user-attachments/assets/c475523b-37af-4e76-9336-bc148c5a1a5d" />
<br><br>


**Analyze Uploaded Image:**
```yaml
action: ai_task.generate_data
data:
  task_name: image analysis
  instructions: Describe what you see in this image
  entity_id: ai_task.custom_openai_ai_tasks
  attachments:
    media_content_id: media-source://media_source/local/my_image.jpeg
    media_content_type: image/jpeg
    metadata:
      title: My uploaded image
      media_class: image
```
<img width="1372" height="1222" alt="image" src="https://github.com/user-attachments/assets/28e81122-463d-4d37-8df9-1a7c0d902f86" />



### Available Models

**Chat models**: Enter any name that your OpenAI or Azure deployment recognizes (e.g., `gpt-4.1`, `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`).

**Image models**: Supported image generation models include `dall-e-2`, `dall-e-3`, `gpt-image-1`, and compatible custom deployments.

## Requirements

- **Home Assistant 2025.10.0 or later** (required for AI Task and AI Image services)
- OpenAI API key, Azure OpenAI key, or custom proxy key
- Endpoint URL for your chosen provider

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Development Approach
<img width="256" height="256" alt="Vibe Coding with GitHub Copilot 256x256" src="https://github.com/user-attachments/assets/bb41d075-6b3e-4f2b-a88e-94b2022b5d4f" />


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues, please report them on the [GitHub Issues page](https://github.com/nuaimat/HA-Custom-OpenAI-AI-tasks/issues).
