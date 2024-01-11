## Для начала

1.
```bash
pip install -r requirements.txt
```
2.
```bash
dvc pull --remote myremote --force
```

Можешь пускать через debug там launch с соотвествующим названием.

## Запуск двух вариантов моделек на видео

Родная:
```bash
python3 commands.py start_yolov8n_pret
```
Твоя:
```bash
python3 commands.py start_yolov8n_alex
```

