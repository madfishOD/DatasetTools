# Mac Compact: перший запуск

Дата: 2026-09-07. CLI captioning працює на поточному Mac; це ще не повне завершення етапу 1M.

## Середовище та результат

- M1 Pro, 16 GPU-ядер, 32 GB shared memory, macOS 26.6.2.
- Python 3.13.5 arm64, PyTorch 2.12.0, Transformers 5.5.4; усі версії — `requirements-mac.lock`.
- Qwen3-VL-2B-Instruct revision `89644892e4d85e24eaac8bacfd4f463576704203`, MPS float16, batch=1, greedy decoding, максимум 320 нових токенів.
- Промпт `prompts/neutral/general.txt`, ліміт довгої сторони 768 px; фактичний вхід 500×500.
- 12/12 captions, 0 зареєстрованих помилок, незмінні джерела/промпти. Image/TXT експорт звірено byte-for-byte.
- Captioning: 4,35–5,59 с на зображення, медіана 5,00 с, сумарно 59,42 с; 64–83 нових токени. Час не включає встановлення, імпорти та завантаження моделі.
- Максимум серед snapshot після зображень: MPS allocated 3,97 GiB, driver 8,72 GiB. Це не безперервно виміряний peak; значення не слід додавати. Memory pressure/swap не вимірювалися.
- Фінальний прогін із `HF_HUB_OFFLINE=1` використовував локальну модель. Це не перевірка встановлення без мережі на чистому Mac.
- 11 локальних unit-тестів пройшли; launcher пройшов `bash -n`.

Раніший прогін із довшим промптом та лімітом 160 токенів показав обрізані й повторювані captions. Додано захист: текст без EOS на ліміті не експортується, raw caption зберігається в JSON. Фінальний короткий промпт дав завершені описи. Вибірковий перегляд показує відповідність основному вмісту; це не розмічений benchmark якості, дрібні деталі потребують ручної перевірки.

## Невеликий відкритий датасет

[Beans від Makerere AI Lab](https://github.com/AI-Lab-Makerere/ibean), [дзеркало Hugging Face](https://huggingface.co/datasets/AI-Lab-Makerere/beans). Ліцензія MIT, copyright AIR Lab Makerere University (2020). Завантажено лише validation split: 133 JPEG, 500×500, 18 490 673 байти зображень (~17,6 MiB); архів 18 504 213 байтів. Дані поза Git.

- Revision: `27aa014ce09b193e1a6f58112d4a66e0eddb69c5`.
- SHA-256 архіву: `90b7aa1c26d91d9afff07a30bbc67a5ea34f1f1397f068d8675be09d7d0c602d`.
- Локальна папка від кореня репозиторію: `projects/test-datasets/beans/validation/`.
- `projects/test-datasets/beans/smoke12/`: перші чотири файли кожної категорії в лексикографічному порядку.
- `SOURCE.json` та `LICENSE.txt` поруч із validation містять походження, список smoke-зразків та ліцензію.

Це фотографії листя трьох категорій. Набір придатний для перевірки pipeline, але не покриває ілюстрації, identity, пози, складні сцени або великі кадри. Для повного етапу 1M потрібна різноманітніша вибірка.

## Відтворення

З кореня репозиторію після створення середовища Mac launcher:

```bash
captioning_tool/runtime/mac/bin/python captioning_tool/fetch_test_dataset.py
./captioning_tool/auto_captioning_tool.command --input projects/test-datasets/beans/smoke12 --check
HF_HUB_OFFLINE=1 ./captioning_tool/auto_captioning_tool.command --input projects/test-datasets/beans/smoke12 --output captioning_tool/outputs/new-mac-run
captioning_tool/runtime/mac/bin/python -m unittest discover -s captioning_tool -p test_tool.py
```

Для першого завантаження моделі заберіть `HF_HUB_OFFLINE=1`. Output має бути новим/порожнім; resume ще немає.

Фінальний результат: `captioning_tool/outputs/mac-beans-smoke12-final/review.html`. Пари — `dataset/data/`, метадані — `regions/*.json` у цьому output. Поле `ready` у поточному exporter означає готовність конфігурації trainer, тому для dataset-only воно `false`, навіть коли всі пари експортовані. Розклад у `training_plan.json` лишається старою евристикою, не персоналізованою порадою.

Наступні перевірки: різні розміри та стилі, довші серії й memory pressure, регіони/SAM Tiny, Windows CUDA regression, потім GUI/resume. Загальний device path не означає, що всі ці комбінації вже перевірені.

Продовження: 40 різноманітних фото COCO128 пройшли технічний captioning smoke, див. [COCO_TESTING.md](COCO_TESTING.md). За уточненням користувача подальші агентські прогони не включають візуального порівняння входів і результатів; перевіряються лише технічні інваріанти та метрики.
