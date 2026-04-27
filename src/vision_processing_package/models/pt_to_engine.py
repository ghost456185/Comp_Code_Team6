#!/usr/bin/env python3

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO


MODEL_PT_PATH = Path(
	'/home/me-cas-jetson04/Comp_Code_Team6/src/vision_processing_package/models/weights.pt'
)
EXPORT_DEVICE = 0


def _collect_export_metadata(engine_path: Path) -> dict:
	metadata = {
		'exported_at_utc': datetime.now(timezone.utc).isoformat(),
		'files': {
			'pt': str(MODEL_PT_PATH),
			'engine': str(engine_path),
		},
		'export': {
			'format': 'engine',
			'device': EXPORT_DEVICE,
			'half': True,
		},
		'software': {
			'torch': torch.__version__,
		},
		'gpu': {},
	}

	try:
		import ultralytics

		metadata['software']['ultralytics'] = ultralytics.__version__
	except Exception:
		pass

	try:
		import tensorrt as trt

		metadata['software']['tensorrt'] = trt.__version__
	except Exception:
		metadata['software']['tensorrt'] = 'unknown'

	if torch.cuda.is_available():
		props = torch.cuda.get_device_properties(EXPORT_DEVICE)
		metadata['gpu'] = {
			'index': EXPORT_DEVICE,
			'name': torch.cuda.get_device_name(EXPORT_DEVICE),
			'capability': f'{props.major}.{props.minor}',
			'total_memory_bytes': int(props.total_memory),
		}
	else:
		metadata['gpu'] = {
			'index': EXPORT_DEVICE,
			'name': 'cuda_unavailable',
		}

	return metadata


def main():
	model = YOLO(str(MODEL_PT_PATH))
	export_result = model.export(format='engine', device=EXPORT_DEVICE, half=True)

	engine_path = Path(str(export_result)) if export_result else MODEL_PT_PATH.with_suffix('.engine')
	if not engine_path.exists():
		engine_path = MODEL_PT_PATH.with_suffix('.engine')

	metadata = _collect_export_metadata(engine_path)
	metadata_path = Path(f'{engine_path}.meta.json')
	metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

	print(f'Engine exported to: {engine_path}')
	print(f'Engine metadata written to: {metadata_path}')


if __name__ == '__main__':
	main()
