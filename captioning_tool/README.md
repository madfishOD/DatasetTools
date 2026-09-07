# auto_captioning_tool

Переносний інструмент для Windows x64: нейтральні captions (JoyCaption), регіони (Qwen3-VL-8B), маски (SAM 2.1 Large) та експорт датасету для OneTrainer. Навчання запускаєте власноруч.

## Запуск

### Експериментальний Mac Compact

Перший MPS-прогін завершено на M1 Pro 32 GB: 12/12 captions без помилок, медіана близько 5 секунд на фото 500×500. Датасет, межі замірів і відтворення описані в [MAC_TESTING.md](MAC_TESTING.md).

Додатково завантажено COCO128 (128 фото, архів ~7 MB) та технічно перевірено captioning на 40 різноманітних фото: 40/40, медіана 5,66 с. Скрипт завантаження — `fetch_coco_test_dataset.py`; команди й результати — [COCO_TESTING.md](COCO_TESTING.md). Це не гарантія семантичної точності captions.

Експериментальний `--stage regions` на Mac: 5/6 відповідей успішно розібрано, SAM Tiny створила 21 маску з перевіреними розмірами й форматом. Одна відповідь обірвалася навіть із `--region-tokens 2048`; регіони поки не можна вважати стабільними для всіх сцен. Default `--region-tokens` — 1000. Невдалий grounding пропускається на SAM-стадії.

Для Apple Silicon додано `auto_captioning_tool.command`: він використовує окреме середовище `runtime/mac`, Python 3.13 і зафіксовані залежності з `requirements-mac.lock`. Потрібен [uv](https://docs.astral.sh/uv/getting-started/installation/); на поточному робочому Mac він уже встановлений. Перший запуск створює середовище та завантажує Qwen3-VL-2B (приблизно 4 GB ваг). Зображення обробляються локально через PyTorch MPS, по одному. GUI ще немає.

Із кореня репозиторію:

```bash
./captioning_tool/auto_captioning_tool.command --input "/path/to/images"
```

Без аргументів launcher запитує input-папку. За замовчуванням: `--profile compact --device mps --stage captions --no-training-config`, загальний промпт `prompts/neutral/general.txt`, до 320 output tokens і довга сторона входу до 768 px. Оригінали не зменшуються; resize застосовується лише до входу моделі. Output має бути новим або порожнім. Параметри можна перевизначати після назви launcher.

`dataset/data/` містить image/TXT пари без конфігурації OneTrainer, `review.html` — зображення й captions. Згенеровані тексти потребують перегляду. Опис, який досяг ліміту токенів без завершення, записується як помилка з raw text у JSON та не експортується. Повний незалежний manifest, персоналізовані поради, затвердження та resume з ROADMAP ще не реалізовані.

Для прямого запуску у вже встановленому середовищі використовуйте `runtime/mac/bin/python auto_captioning_tool.py` з потрібними параметрами. `--device` підтримує `auto`, `cuda`, `mps`, `cpu`; автоматичний вибір не переходить мовчки на CPU. Mac за замовчуванням використовує float16; `--dtype float32` доступний для діагностики. Прихований MPS CPU fallback відхиляється. `--check` тепер перевіряє також доступність вибраного пристрою й dtype, тому потребує PyTorch; замірів inference, пікової пам’яті й повного аудиту ресурсів він не виконує.

У JSON зразків записуються модель/revision, device/dtype, час captioning, розмір входу, кількість токенів і snapshot пам’яті. MPS snapshot — значення в момент вимірювання, не пікова пам’ять. CUDA-шлях і новий Compact на Windows потребують окремого GPU-прогону; Mac-регіони та SAM пройшли лише обмежений технічний smoke, описаний вище.

### Windows

Розпакуйте весь каталог у доступне для запису місце. Запустіть `auto_captioning_tool.bat`. Перший запуск завантажує Python, бібліотеки та відсутні моделі. Вкажіть шлях до зображень, output-папку, три TXT-файли, сімейство й базову модель навчання. Enter залишає показане значення. Output має бути новим або порожнім і знаходитися поза input.

Обробляються всі PNG, JPG/JPEG, WebP та BMP, включно з підпапками. `--no-recursive` вимикає підпапки. Анімовані зображення записуються як помилки: спочатку експортуйте окремі кадри. Відео, TIFF, HEIC та RAW у цій версії не підтримуються. Junction-папки й символічні посилання на папки не обходяться.

```bat
auto_captioning_tool.bat --input "E:\Dataset" --output "E:\Captioned" --prompt "E:\Prompts\instruction.txt" --system-prompt "E:\Prompts\system.txt" --region-prompt "E:\Prompts\regions.txt" --family qwen --base-model "Qwen/Qwen-Image"
```

Приклади параметрів:

```bat
auto_captioning_tool.bat --input "E:\Dataset" --check
auto_captioning_tool.bat --download-models
auto_captioning_tool.bat --input "E:\Dataset" --output "E:\SDXL_run" --family sdxl --stage captions --epochs 20
auto_captioning_tool.bat --input "E:\Dataset" --output "E:\Flux_run" --family flux --target-steps 2000 --rank 16 --learning-rate 0.0001
```

`--check` перевіряє шляхи та наявність моделей без inference або їх завантаження. Сам BAT спочатку встановлює відсутнє середовище. Для суто read-only перевірки у вже наявному Python запускайте `python auto_captioning_tool.py ... --check`.

## Перенесення на інший ПК

Копіюйте весь каталог, включно з `runtime/` та `models/`, щоб не завантажувати їх повторно. Шляхи до середовища й кешу відносні. Python не потребує системного встановлення або прав адміністратора. Порожній кеш заповнюється автоматично з Hugging Face; зафіксовані revisions моделей зазначені в `engine.py`. Незавершене завантаження можна повторити запуском тієї самої команди.

За потреби вкажіть інший кеш через `--models`. Це моделі **captioning**, а не базова модель навчання. `--base-model` приймає repository ID або локальний шлях базової моделі **обраного сімейства**. FLUX тут означає FLUX.1-dev; Qwen означає Qwen Image. Інші архітектури не стають сумісними просто через заміну шляху.

Потрібні NVIDIA GPU із BF16, сумісний із CUDA 13 драйвер, Windows x64 та Microsoft Visual C++ runtime. Поточний профіль розрахований приблизно на 24+ GB VRAM; перевірений на RTX 5090 32 GB. Моделі займають десятки GB. Передбачте щонайменше близько 50 GB для кешу й бібліотек, плюс копії датасету та checkpoints. Драйвер інструмент не встановлює. Повністю автономний запуск можливий після завантаження всього кешу.

Режим повторного використання існуючого Python: змінна `AUTO_CAPTIONING_PYTHON` містить шлях до python.exe; за потреби додайте `--packages` до команди. За замовчуванням використовується тільки локальний runtime. `AUTO_CAPTIONING_NO_PAUSE=1` вимикає фінальну паузу BAT для автоматизації.

## Output

- `images/`: незмінені копії всіх вхідних зображень.
- `captions/`, `regions/`, `masks/`, `previews/`, `grounding_raw/`, `review.html`: формат попереднього pipeline.
- `prompts/`: точні копії вибраних TXT; оригінали не редагуються.
- `source_manifest.json`: відповідність вихідних шляхів новим назвам та SHA-256. Однакові stems отримують стабільні унікальні ідентифікатори.
- `errors/`, `validation.json`: помилки й перевірка незмінності джерел та промптів.
- `onetrainer/data/`: лише успішні image + TXT пари; обидва файли копіюються byte-for-byte.
- `onetrainer/train.json`, `concepts.json`, `samples.json`, `training_plan.json`: конфігурації та обґрунтування початкового розкладу.

У OneTrainer завантажте `onetrainer/train.json`, перевірте вкладки Model/Concepts/Training та запустіть навчання. Базова модель має включати необхідні encoder/VAE; repository ID може потребувати завантаження або авторизації. Скрипт не завантажує великі базові моделі навчання самостійно.

Qwen, SDXL і FLUX мають шаблони поточної схеми OneTrainer. WAN експортує тільки датасет і пояснення в `training_plan.json`: перевірений OneTrainer не підтримує WAN, тому сумісний `train.json` створити неможливо. За відсутності успішних captions `train.json` також не створюється. `--stage regions` не робить training export.

Розклад: N = кількість успішних пар; ціль = clamp(20*N, 500, 3000) кроків; епохи = ceil(ціль/N). Batch=1, accumulation=1, repeats=1. Наприклад, 117 пар -> 20 епох -> 2340 кроків. `--epochs` має пріоритет над `--target-steps`. Для дуже малих наборів високе повторення може спричинити перенавчання. LR=1e-4 і rank/alpha=16 — початкові значення, не автоматично оптимальні. Розмір датасету сам по собі не визначає оптимальні LR/rank/resolution. За замовчуванням: Qwen/FLUX 768, SDXL 1024; доступний `--resolution`.

Маски потребують перегляду: overlap та SAM score — лише сигнали для перевірки. Вони не є готовим conditioning для кількох персонажів. `masked_training=false`; автоматичний flip вимкнений через згадки лівого/правого боку у captions. Checkpoints зберігаються щoепохи.

Після переміщення готового output оновіть абсолютні шляхи конфігурації:

```bat
auto_captioning_tool.bat --rebase-output "F:\MovedOutput"
```

Це змінює лише згенеровані шляхи OneTrainer. Локальний `base_model_name` перевірте окремо. Зображення, captions і prompt TXT не змінюються.

## Режим описів і перевірки

Exit code 0 означає завершення без зареєстрованих помилок, 1 — помилки; часткові результати залишаються для перегляду. Новий запуск потребує нової/порожньої output-папки. Автоматичного resume обробки зображень поки немає.

Перевірено: 6 локальних тестів; round-trip JSON через класи OneTrainer для всіх трьох сімейств; повний GPU pipeline на двох штучних зображеннях: 2 captions, 3 маски, 0 помилок, незмінні хеші. Перевірено завантаження embedded Python/pip і BAT preflight. GPU тест використовував уже встановлені бібліотеки тих самих версій. Повне встановлення всіх CUDA-бібліотек на чистому іншому ПК і навчання не запускалися.

Шаблони похідні від [OneTrainer](https://github.com/Nerogar/OneTrainer), локальна схема TrainConfig v11, ConceptConfig v2, перевірено 2026-09-06. Див. `templates/OneTrainer-LICENSE.txt`. Профіль Qwen очищено від застарілого верхнього поля weight_dtype; тип стандартного cloud.port нормалізовано серіалізатором OneTrainer.
