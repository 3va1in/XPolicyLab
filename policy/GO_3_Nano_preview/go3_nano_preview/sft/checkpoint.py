import json

def save_checkpoint(model, output, step):
    folder = output / f'step_{step}'
    folder.mkdir(parents=True, exist_ok=False)
    model.model.vgen.save_pretrained(folder, safe_serialization=False, max_shard_size='100GB')
    model.model.ae.save_pretrained(folder / 'action_model', safe_serialization=False, max_shard_size='100GB')
    config = folder / 'config.json'
    data = json.loads(config.read_text())
    if 'vlm_model_id' in data:
        data['vlm_model_id'] = '${VLM_MODEL_PATH}'
    config.write_text(json.dumps(data, indent=2) + '\n')
    return folder
