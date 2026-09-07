# Базовий GUI captioning

Мова всього інтерфейсу застосунку — англійська, незалежно від мови спілкування з розробником.

Запуск на поточному Apple Silicon Mac із кореня репозиторію:

```bash
./captioning_tool/gui.command --project projects/coco40-resumable
```

Або двічі відкрийте `gui.command` і виберіть папку проєкту. На Windows передбачений `gui.bat`; його запуск і пакування ще потребують окремої перевірки. Qt Widgets встановлюється з `requirements-gui.txt` (PySide6-Essentials/shiboken6 6.10.3). Mac launcher синхронізує також ці залежності, щоб наступний CLI-запуск їх не видаляв.

## Робочий цикл

1. **New project…** відкриває одне вікно з двома полями: **Input** — наявна папка вихідних зображень, **Output** — окрема порожня або нова папка проєкту та результатів. Для нової Output-папки можна ввести повний шлях. Кнопка **Create and import** показана під обома полями. Зображення й сусідні TXT імпортуються без inference. Нові проєкти використовують Compact, auto device, captions і незалежний експорт.
2. **Run / Resume**: запускає CLI в окремому процесі зі збереженими налаштуваннями. Для нового проєкту відсутні моделі завантажуються автоматично.
3. **Stop**: чекає завершення поточної операції та збереження checkpoint. Завантаження/ініціалізація моделі, імпорт і експорт можуть потребувати часу до зупинки. Закриття вікна під час роботи запитує Stop і залишає вікно відкритим; після зупинки закрийте повторно.
4. Select a sample, відредагуйте caption і натисніть **Save edit** або **Save and approve**. Збереження без затвердження виключає цю версію з наступного експорту. Порожній текст затвердити не можна.
5. **Export approved** створює незалежний image/TXT пакет із manifest. **Open export folder** показує його папку. Це дія без inference; OneTrainer-конфігурація через цю кнопку не створюється.

Фільтри показують усі, неперевірені, затверджені або помилкові зразки. **Regenerate selected** явно замінює caption одного зразка, зберігаючи попередній текст в історії та скидаючи затвердження. **Refresh** перечитує зовнішні TXT-правки. Якщо файл змінився після відкриття в редакторі, збереження відхиляється: скопіюйте незбережений текст, оновіть проєкт і узгодьте правки.

На переході з незбереженим текстом доступні Save/Discard/Cancel. Системні скорочення Open/Save працюють через Command на Mac і Ctrl на Windows. Під час worker-операції редагування та перемикання проєкту заблоковані, вікно й журнал залишаються активними. Прев’ю завантажується лише для вибраного зразка; весь датасет у пам’ять не завантажується.

## Перевірки й межі першої версії

- 38 автоматичних тестів разом із попередніми тестами pipeline: збереження/затвердження, незмінність імпортованих байтів, конфлікт зовнішніх правок, скасування переходу, реальний subprocess експорту, повторний запуск, аварія worker та кооперативний Stop.
- Реальний headless Qt/MPS smoke на двох фото: Stop під час першої операції, checkpoint першого caption, Resume лише другого; SHA-256 та mtime першого TXT незмінні. Наступний запуск мав нуль запланованих стадій.
- Візуальні й семантичні порівняння результатів не виконувалися. Qt-тести вимикають прев’ю.
- На старті GUI перевіряє хеші файлів синхронно; для великих датасетів відкриття ще потрібно винести з UI-потоку. Поточна перевірка — малий датасет.
- **Project settings…** дозволяє задати мету навчання й окремо вибрати моделі для кожної стадії; **Training advice…** показує поради до експорту. Деталі: [ADVICE_MODELS.md](ADVICE_MODELS.md). Редактор промптів, thumbnails-галерея, crops і редактор масок залишаються наступними частинами плану. GUI не означає завершення всього етапу 3.

Worker використовує [Qt QProcess](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QProcess.html) і той самий `auto_captioning_tool.py`. CLI передає структуровані `DATASET_EVENT` через stdout; Stop задається окремим тимчасовим файлом для конкретного запуску. JSON checkpoints лишаються джерелом стану після аварії. Приватні аргументи `--events` і `--stop-file` не змінюють збережені параметри проєкту.

## Hugging Face authentication

Use **Hugging Face token…** in the main toolbar to enter a read token. Input is masked. **Remember token in system credential storage** is enabled by default: the app saves its token in macOS Keychain or Windows Credential Manager and loads it automatically on the next launch. The operating system may request permission to access its credential store. If access fails, the app reports the problem and does not fall back to a plaintext file.

Uncheck **Remember token…** for session-only use; saving this choice removes any previously stored app token. Leave the field empty and save to remove the app token entirely. Existing environment/Hugging Face login settings still apply; removing the app token does not log out other applications.

The token is passed to the next CLI worker through `HF_TOKEN`, never through command-line arguments, project metadata, exports or QSettings. Token-shaped values are redacted from the GUI worker log. The button is available during processing, but changes cannot modify an already-running download. After it stops, use **Run / Resume** to apply the token.

The dialog links to the Hugging Face token page. Local validation checks only format; Hugging Face checks permissions during download. Public models can still download without a token. Persistent storage is local to the OS user account and is not copied when a project moves to another machine.

Credential lifecycle tests use a fake store and fake tokens; they do not access personal credentials or validate a live Hugging Face account. Windows credential storage still requires a Windows integration smoke.

Implementation references: [Hugging Face environment variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables), [keyring native credential stores](https://keyring.readthedocs.io/en/stable/).
