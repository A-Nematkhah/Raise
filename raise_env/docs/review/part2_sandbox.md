<div dir="rtl">

# بررسی بخش ۲: Sandbox و اعتبارسنجی کد پاداش

**دامنه:**

- `raise_core/sandbox/ast_policy.py`، `config.py`، `runtime.py`، `validator.py`
- `domains/highway/reward_checks.py`، `sandbox_fields.py`، `watchdog.py`
- `raise_core/rejection_log.py`
- بخش نرمال‌سازی کد در `raise_core/llm.py` (`normalize_to_compute_reward` و `strip_redundant_scalar_subscripts`)

نقاط اتصال هم بررسی شد: `env_wrapper.step` (محل صدا زدن پاداش در آموزش)، `stage1.py` (Score1)، `raise_loop/parallel_label.py` (Stage II) و `validate.py` (Stage III).

**روش:** کد خط به خط خوانده شد. بعد با اسکریپت‌های آزمایشی، ده‌ها کد پاداش (سالم، مخرب، کند، سرریز، با state سراسری و ...) از همان مسیر واقعی `HighwayRewardValidator` عبور داده شد و رفتار زمان اجرا هم اندازه‌گیری شد. اعداد بخش «شواهد» از همین اجراهاست.

> بخش‌های ۱ تا ۵ وضعیت کد **قبل از اصلاح** را توصیف می‌کنند. اینکه هر یافته چطور اصلاح شد و نتیجه‌ی آزمایش‌های بعد از اصلاح، در بخش ۶ آمده است.

**سطح‌بندی یافته‌ها:**

- **H (بالا):** روی اعتبار نتایج، پایداری pipeline یا بازده حلقه‌ی LLM اثر مستقیم دارد.
- **M (متوسط):** باگ واقعی که در شرایط خاص رخ می‌دهد، یا هزینه‌ی محاسباتی قابل‌توجه.
- **L (پایین):** تمیزکاری، دسته‌بندی لاگ یا کد مرده.

</div>

---

<div dir="rtl">

## ۱. خلاصه‌ی معماری

هر کدی که LLM تولید می‌کند، قبل از ورود به جمعیت از این زنجیره می‌گذرد:

</div>

```text
LLM text
  -> normalize_to_compute_reward              llm.py
       rename to compute_reward, add memory arg,
       strip state.vx[0]-style scalar subscripts (ast.unparse)
  -> HighwayRewardValidator.validate_code     reward_checks.py
       unicode normalization
       check_target_speed_literals / warn_inline_speed_literals
  -> RewardValidator.validate_code            sandbox/validator.py
       parse_reward_code                      length <= 20000, ast.parse
       check_structure (_PolicyVisitor)       ast_policy.py
            no import / while / comprehensions / class / with / global ...
            attributes: no "_" prefix, must be on HIGHWAY allowlist + dict methods
            names: no FORBIDDEN_NAME_IDS, no "__" prefix
       check_interface                        exactly one compute_reward(state, memory)
       check_top_level_constants              only literal constants at module level
       smoke_test_compute                     spawned child, 5 s timeout, 12 states
            finite float (or (float, dict) with <= 12 components)
       compile_compute_reward                 exec in parent with _SAFE_BUILTINS + math
  -> SandboxedReward                          per-episode memory, cleared on reset()
```

<div dir="rtl">

در زمان آموزش، `env_wrapper.step` در هر step تابع `SandboxedReward.compute(state)` را **داخل همان process** و **بدون timeout** صدا می‌زند. خطاها به `RewardSandboxError` تبدیل می‌شوند. تنها محافظ زمانی، `StallWatchdog` است که فقط وقتی هیچ step ای به مدت ۳۰۰ ثانیه انجام نشود فعال می‌شود.

**نتیجه‌ی کلی:** هسته‌ی امنیتی sandbox محکم است و هیچ‌کدام از تلاش‌های فرار از sandbox موفق نشد (بخش ۴). مشکلات اصلی در **قرارداد زمان اجرا** هستند: چیزهایی که smoke test روی ۱۲ state نمی‌بیند (کندی وابسته به داده، state سراسری، بزرگی پاداش) و رفتار pipeline وقتی پاداش بعد از پذیرش خطا می‌دهد. یک ناسازگاری جدی هم بین پیام‌ها و allowlist وجود دارد: `math` تزریق می‌شود ولی عملاً قابل استفاده نیست.

</div>

---

<div dir="rtl">

## ۲. یافته‌ها

### H1. ماژول `math` تزریق می‌شود ولی هر استفاده‌ای از آن رد می‌شود

**محل:** `ast_policy.py` خطوط ۲۹ و ۴۳ (پیام import) و خط ۱۲۵ (allowlist)، `runtime.py` (`compile_compute_reward` که `math` را به namespace اضافه می‌کند)، `prompts.py` خطوط ۱۰۶ تا ۱۰۷ و ۱۴۹.

**مشکل:**

- `compile_compute_reward` نام `math` را در namespace قرار می‌دهد و پیام خطای import صریحاً می‌گوید: «math is pre-injected as name `math`».
- ولی `_PolicyVisitor.visit_Attribute` هر attribute ای را که روی allowlist دامنه نباشد رد می‌کند. `sqrt`، `exp`، `tanh`، `log`، `pi` و بقیه روی allowlist نیستند.
- نتیجه: `math.exp(...)`، `math.tanh(...)` و حتی `math.pi` همه رد می‌شوند، با پیام گمراه‌کننده‌ی «attribute 'exp' is not on the domain allowlist (hallucinated field?)».
- پرامپت فقط می‌گوید «import math ننویس، از `** 0.5` استفاده کن». هیچ‌جا نمی‌گوید `math.*` ممنوع است، و پیام خطای import هم LLM را به استفاده از `math` تشویق می‌کند.

**اثر:**

- `exp`، `tanh` و `log` رایج‌ترین ابزارهای LLM برای شکل‌دادن پاداش هستند (مثلاً `math.exp(-gap / 10)`). هر کاندیدی که از آن‌ها استفاده کند در Gen0 و در جهش‌ها رد می‌شود.
- هزینه: یک فراخوانی LLM هدررفته و یک دور retry به ازای هر رد. در نسل‌های بعدی، بازخورد خطا («hallucinated field») LLM را به اشتباه می‌اندازد، چون او فکر می‌کند فیلد state را اشتباه نوشته است.
- بیان‌پذیری پاداش هم کم می‌شود: `tanh` و `exp` را فقط با تقریب‌های دستی می‌توان ساخت.

**شواهد:** همه‌ی این موارد از مسیر واقعی validator رد شدند:

</div>

```text
math.sqrt(x)      REJECT  attribute 'sqrt' is not on the domain allowlist (hallucinated field?)
math.exp(-x)      REJECT  attribute 'exp' ...
math.tanh(x)      REJECT  attribute 'tanh' ...
math.log(1 + x)   REJECT  attribute 'log' ...
math.pi           REJECT  attribute 'pi' ...
x ** 0.5          ACCEPT
```

<div dir="rtl">

**پیشنهاد اصلاح:**

- **گزینه‌ی الف (پیشنهادی):** در `visit_Attribute`، اگر `node.value` یک `Name` با `id == "math"` بود، attribute را با یک allowlist جداگانه چک کن:
  `sqrt, exp, log, log1p, log10, tanh, sin, cos, atan2, hypot, fabs, floor, ceil, copysign, isfinite, pi, e, tau, inf`.
  این توابع خالص هستند و راه فراری ندارند. `math.exp(1000)` خطای `OverflowError` می‌دهد که مثل هر خطای زمان اجرای دیگر مدیریت می‌شود (H2 را ببینید). یک خط هم در پرامپت اضافه شود که دقیقاً همین توابع مجازند.
- **گزینه‌ی ب:** `math` را از namespace حذف کن، پیام import را اصلاح کن و در پرامپت صریحاً بنویس «`math.*` ممنوع است».

</div>

---

<div dir="rtl">

### H2. خطای زمان اجرای پاداش بعد از پذیرش: Stage III کل pipeline را متوقف می‌کند

**محل:** `runtime.py` خطوط ۳۹۶ تا ۴۱۰ (`SandboxedReward.compute`)، `validate.py` خطوط ۳۰۰ تا ۳۰۹ (حلقه‌ی Stage III)، `parallel_label.py` خطوط ۱۱۰ تا ۱۱۹.

**مشکل:**

- smoke test فقط ۱۲ state را امتحان می‌کند. کدی که روی آن‌ها سالم است، می‌تواند در آموزش روی stateهای دیگر خطا بدهد: تقسیم بر صفر وقتی `gap == 0`، اندیس خارج از محدوده وقتی `others` خالی است، سرریز در `math.exp` یا `**`، یا خروجی `inf`/`nan`.
- `SandboxedReward.compute` هر خطا را به `RewardSandboxError` تبدیل می‌کند و دوباره raise می‌کند (رفتار درست: fail-fast).
- **Stage II:** `_safe_label` خطا را می‌گیرد و کاندید `failed` می‌شود. درست است.
- **Stage III:** حلقه‌ی `for i in range(start, n)` در `validate.py` هیچ `try/except` ای ندارد. یک خطا در هر کاندید، کل اجرای Stage III را متوقف می‌کند.
- چون checkpoint فقط **بعد** از هر کاندید ذخیره می‌شود (`next_index=i + 1`)، resume دوباره از همان کاندید شروع می‌کند و دوباره crash می‌کند. pipeline هرگز تمام نمی‌شود.

**اثر:** یک کاندید نهایی که در آموزش طولانی Stage III (با step بیشتر و stateهای متنوع‌تر) به یک حالت مرزی برخورد کند، کل اجرای پایان‌نامه را قفل می‌کند. احتمالش کم نیست، چون Stage III طولانی‌ترین آموزش را دارد.

**پیشنهاد اصلاح:** در حلقه‌ی Stage III، `train_and_eval` را در `try/except Exception` بگذار. کاندید را با `status="failed"` و دلیل خطا ثبت کن (fitness برابر `-inf`)، checkpoint را جلو ببر و ادامه بده. انتخاب «بهترین» کاندید باید کاندیدهای failed را نادیده بگیرد. این بخش در اصل به بخش ۸ مربوط است، ولی ریشه‌اش قرارداد زمان اجرای sandbox است و اصلاحش کوچک است.

</div>

---

<div dir="rtl">

### M1. کندی وابسته به داده شناسایی نمی‌شود

**محل:** `runtime.py` خط ۳۹۶ (`compute` بدون اندازه‌گیری زمان)، `watchdog.py` خط ۲۸ (`DEFAULT_STALL_SECONDS = 300`).

**مشکل:**

- timeout پنج‌ثانیه‌ای فقط در smoke test اعمال می‌شود. حلقه‌ای که تعداد تکرارش به state وابسته است (مثلاً `for i in range(int(ego.x))`)، روی stateهای smoke سریع است و پذیرفته می‌شود.
- در آموزش، `ego.x` تا حدود ۱۰۰۰ متر بالا می‌رود و هر step کند می‌شود.
- watchdog فقط وقتی فعال می‌شود که **هیچ** step ای به مدت ۳۰۰ ثانیه انجام نشود. stepهای کند ولی پیوسته را نمی‌بیند.

**شواهد:**

</div>

```text
for i in range(int(ego.x) * 2000): ...      ACCEPT (smoke OK)
compute() at ego.x = 1000:                   0.46 s per step
12k PPO steps:                               ~1.5 h for ONE candidate (normal: ~1-2 min)
StallWatchdog:                               never fires (a step completes every 0.46 s)
```

<div dir="rtl">

**اثر:** یک کاندید می‌تواند یک worker را ساعت‌ها اشغال کند. در اجرای موازی Stage II، کل نسل منتظر همین یک کاندید می‌ماند.

**پیشنهاد اصلاح:**

- در `SandboxedReward.compute`، زمان هر فراخوانی با `time.perf_counter()` اندازه‌گیری شود. اگر یک فراخوانی از سقف مشخصی بیشتر شد (مثلاً ۲۵۰ میلی‌ثانیه)، یا میانگین متحرک ۱۰۰ فراخوانی اخیر از مثلاً ۱۰ میلی‌ثانیه بیشتر شد، `RewardSandboxError("reward too slow ...")` raise شود. Stage II آن را مثل هر خطای دیگر `failed` می‌کند (و با اصلاح H2، Stage III هم).
- هزینه‌ی سربار: دو فراخوانی `perf_counter` در هر step، که قابل‌اغماض است.
- مقدار سقف‌ها در `SandboxConfig` قرار بگیرد تا قابل تنظیم باشد.
- اختیاری: smoke worker هم زمان هر state را بسنجد و اگر یک فراخوانی بیش از مثلاً ۵۰ میلی‌ثانیه طول کشید، رد کند.

</div>

---

<div dir="rtl">

### M2. ثابت‌های سراسری تغییرپذیر بین اپیزودها نشت می‌کنند

**محل:** `ast_policy.py` خطوط ۲۲۸ تا ۲۵۰ (`check_top_level_constants` که `ast.List`، `ast.Dict` و `ast.Set` را مجاز می‌داند).

**مشکل:**

- قرارداد sandbox این است که state بین stepها فقط در `memory` نگه داشته شود و `memory` در `reset()` پاک شود.
- ولی یک لیست یا dict سراسری (`HIST = []`) پذیرفته می‌شود و `HIST.append(...)` هم مجاز است، چون `append` در `SAFE_METHOD_ATTRIBUTES` است.
- این شیء در namespace تابع زندگی می‌کند و `reset()` به آن دست نمی‌زند. پس بین اپیزودها باقی می‌ماند و بدون حد رشد می‌کند.

**شواهد:**

</div>

```text
HIST = []
def compute_reward(state, memory):
    HIST.append(1.0)
    return float(len(HIST))

validation:                        ACCEPT
episode 0 (after reset()):         [1.0, 2.0, 3.0, 4.0, 5.0]
episode 1 (after reset()):         [6.0, 7.0, 8.0, 9.0, 10.0]
episode 2 (after reset()):         [11.0, 12.0, 13.0, 14.0, 15.0]
```

<div dir="rtl">

**اثر:**

- پاداش **غیرایستا** می‌شود: یک state یکسان در اول و آخر آموزش پاداش متفاوت می‌گیرد. PPO چنین سیگنالی را درست یاد نمی‌گیرد و مقایسه‌ی کاندیدها بی‌معنی می‌شود.
- اگر کد روی `HIST` حلقه بزند، هزینه‌ی هر step با زمان رشد می‌کند و به M1 تبدیل می‌شود.
- هر VecEnv worker یک کپی جدا دارد (unpickle دوباره کامپایل می‌کند)، پس رفتار به تعداد envها هم وابسته می‌شود.
- LLM به‌طور طبیعی چنین الگویی می‌نویسد، مثلاً `PREV = {}` برای ذخیره‌ی مقدار قبلی.

**پیشنهاد اصلاح:** در `check_top_level_constants` فقط ثابت‌های **تغییرناپذیر** مجاز باشند: `Constant`، `Tuple` (با عناصر تغییرناپذیر)، `UnaryOp`، `BinOp`، `Name`. اگر `List`، `Dict` یا `Set` در سطح ماژول آمد، با پیامی شفاف رد شود: «mutable module-level state is forbidden; use the memory dict (cleared every episode)». یک تست هم اضافه شود.

</div>

---

<div dir="rtl">

### M3. هیچ کنترلی روی بزرگی پاداش وجود ندارد

**محل:** `runtime.py` (`require_finite_float` و `unpack_reward_return`).

**مشکل:**

- smoke test فقط متناهی بودن خروجی را چک می‌کند. پاداشی که `1e15 * progress` برمی‌گرداند پذیرفته می‌شود.
- PPO در این پروژه بدون نرمال‌سازی پاداش (`VecNormalize`) اجرا می‌شود. پاداش‌های بسیار بزرگ value loss را منفجر می‌کنند و gradient clipping فقط بخشی از آسیب را جبران می‌کند. نتیجه، آموزش بی‌ثمر و fitness پایین است. این کاندید در عمل به‌خاطر مقیاس عددی می‌بازد، نه به‌خاطر ایده‌ی پاداشش.
- پاداش ثابت (`return 0.0`) هم در sandbox پذیرفته می‌شود. Score1 آن را با `reject_reason="no_signal"` رد می‌کند (`stage1.py` خطوط ۵۵۱ تا ۵۶۱)، **ولی** `_needs_training` در `runner.py` (خط ۱۴۷) پرچم `score1_rejected` را نگاه نمی‌کند. پس این کاندید باز هم یک دور کامل PPO آموزش داده می‌شود. این مورد به بخش‌های ۴ و ۷ مربوط است و اینجا فقط ثبت می‌شود.

**شواهد:**

</div>

```text
return 1e15 * float(state.progress)     ACCEPT
return 0.0                              ACCEPT (sandbox) -> Score1 no_signal -> still PPO-trained
```

<div dir="rtl">

**پیشنهاد اصلاح:**

- در smoke worker، اگر `|total|` روی هر state از سقفی مثل `1e4` بیشتر شد، کاندید با پیام «reward magnitude too large; keep per-step rewards within ±10^4» رد شود. سقف در `SandboxConfig` قرار بگیرد. جریمه‌های پایانی رایج LLM (مثلاً ‎-100 یا ‎-1000) زیر این سقف هستند.
- جایگزین نرم‌تر: پاداش در `env_wrapper` clip شود. ولی این مقیاس واقعی پاداش را پنهان می‌کند و بهتر است فقط به‌عنوان لایه‌ی دوم استفاده شود.
- پاداش ثابت را Score1 مدیریت می‌کند. فقط باید runner کاندیدهای `score1_rejected` را آموزش ندهد (در بخش ۷ بررسی می‌شود).

</div>

---

<div dir="rtl">

### L1. دسته‌بندی دلیل رد در `rejection_log` ناقص است

**محل:** `raise_core/rejection_log.py`.

**مشکل:**

- رد شدن به‌خاطر allowlist («not on the domain allowlist»)، که احتمالاً رایج‌ترین دلیل رد کد LLM است، در دسته‌ی `other` می‌افتد.
- همه‌ی ساختارهای ممنوع (comprehension، while، class، with و ...) چون عبارت «is forbidden» دارند، در دسته‌ی `forbidden_name` می‌افتند.
- شرط `"timeout" in text` **قبل** از `"runtime error"` چک می‌شود. پیام خطای زمان اجرایی که نام فیلد `timeout` را دارد (مثلاً `runtime error ... 'timeout'`) به اشتباه در دسته‌ی `timeout` قرار می‌گیرد.

**اثر:** آمار رد شدن در Gen0 (که برای تحلیل پایان‌نامه مفید است) دقیق نیست و مهم‌ترین دلیل رد دیده نمی‌شود.

**پیشنهاد اصلاح:** دسته‌های `not_on_allowlist` و `forbidden_construct` اضافه شوند. برای timeout، الگوی دقیق پیام (`execution exceeded timeout of`) چک شود، نه هر رخداد کلمه‌ی timeout.

</div>

---

<div dir="rtl">

### L2. `_SCALAR_FIELD_ATTRS` هنوز نام‌های CrowdNav را دارد

**محل:** `llm.py` خطوط ۶۸۹ تا ۷۳۸.

**مشکل:**

- `strip_redundant_scalar_subscripts` قرار است الگوهایی مثل `ego.speed[0]` را به `ego.speed` تبدیل کند (اشتباه رایج LLM).
- ولی مجموعه‌ی `_SCALAR_FIELD_ATTRS` شامل `px`، `py`، `radius`، `gx` و دیگر نام‌های CrowdNav است. از فیلدهای اسکالر highway فقط `vx`، `vy`، `collision`، `timeout` و `global_time` در آن هستند. `speed`، `progress`، `x`، `y`، `heading`، `lane_index` و `off_road` در آن نیستند.
- در نتیجه `ego.speed[0]` حذف نمی‌شود و smoke با `TypeError` رد می‌کند.

**نکته‌ی مثبت:** `state.action` در این مجموعه نیست، پس `state.action[0]` در حالت پیوسته دست نمی‌خورد (درست است).

**پیشنهاد اصلاح:** مجموعه با فیلدهای اسکالر واقعی highway جایگزین شود و نام‌های CrowdNav حذف شوند.

</div>

---

<div dir="rtl">

### L3. مسیر fallback در `smoke_test_compute` مرده و خراب است

**محل:** `runtime.py` خطوط ۳۴۲ تا ۳۵۸.

**مشکل:** وقتی `source_code` داده نشود، `smoke_test_compute` یک تابع محلی `_run` را به `run_with_timeout` می‌دهد. `run_with_timeout` از `spawn` استفاده می‌کند و تابع محلی قابل pickle نیست، پس این مسیر همیشه با خطای pickle شکست می‌خورد. validator همیشه `source_code` را می‌فرستد، پس این مسیر هیچ‌وقت در عمل اجرا نمی‌شود. تنها استفاده‌کننده‌ی `run_with_timeout` هم یک تست است.

**پیشنهاد اصلاح:** `source_code` اجباری شود و مسیر fallback حذف شود. `run_with_timeout` هم اگر فقط در تست استفاده می‌شود حذف شود، یا فقط به‌عنوان ابزار عمومی بماند.

</div>

---

<div dir="rtl">

### L4. جزئیات فرایند فرزند و smoke

- **`proc.close()`** بعد از `join` صدا زده نمی‌شود و `Queue` هم بسته نمی‌شود. در یک اجرای طولانی با صدها کاندید، handleها تا GC باز می‌مانند. اثرش کم است ولی اصلاحش یک خط است.
- **ترتیب `join` و بعد `get`:** اگر payload صف بزرگ باشد، فرزند ممکن است روی flush صف گیر کند و `join` منقضی شود. payloadها اینجا کوچک‌اند (یک رشته‌ی خطا)، پس در عمل مشکلی نیست. ولی ترتیب امن‌تر `get(timeout)` و بعد `join` است.
- **smoke با حافظه‌ی مشترک:** هر ۱۲ state با یک `memory` مشترک و بدون reset اجرا می‌شوند، در حالی که این stateها از اپیزودهای نامرتبط هستند. فراخوانی اول مسیر «memory خالی» و بقیه مسیر «memory پر» را پوشش می‌دهند، که کافی است. ولی مثلاً `dt = global_time - memory["t"]` ممکن است در smoke منفی یا صفر شود، در حالی که در آموزش هرگز این‌طور نیست، و کاندید سالم بی‌دلیل رد شود. این مورد اندازه‌گیری نشد و فقط یک ریسک بالقوه است.
- **زمان spawn داخل بودجه‌ی ۵ ثانیه حساب می‌شود.** کل اعتبارسنجی حدود ۰٫۲۳ ثانیه طول می‌کشد، پس حاشیه کافی است.

</div>

---

<div dir="rtl">

## ۳. موارد خارج از دامنه که اینجا پیدا شد

| مورد | بخش مقصد |
|---|---|
| کاندید `score1_rejected` باز هم با PPO آموزش داده می‌شود (`_needs_training`) | ۴ و ۷ |
| Stage III بدون `try/except` (جزئیات در H2) | ۸ |

</div>

---

<div dir="rtl">

## ۴. مواردی که بررسی و تأیید شد

- **مقاومت در برابر فرار از sandbox:** همه‌ی این تلاش‌ها رد شدند: دسترسی به dunderها (`().__class__`، `memory.__class__`)، `.format` و f-string با dunder، `type`، `getattr`، `__builtins__`، `__import__`، `import`، `while`، `global`. `exec` در process والد فقط تعریف تابع را اجرا می‌کند، چون `check_top_level_constants` کد سطح ماژول را محدود می‌کند.
- **بازگشت بی‌پایان:** در smoke با `RecursionError` رد می‌شود.
- **حلقه‌ی بی‌پایان:** فرزند بعد از ۵ ثانیه kill می‌شود و هیچ process ای جا نمی‌ماند (تست موجود `test_sandbox_timeout_kill`).
- **خروجی‌های نامتناهی:** `nan`، `inf`، `bool`، `None` و رشته رد می‌شوند. dict اجزا با بیش از ۱۲ کلید یا مقادیر نامتناهی هم رد می‌شود.
- **سرریز در محدوده‌ی smoke:** `speed ** ego.x` رد شد، چون stateهای smoke شامل `ego.x = 573` هستند. سرریز در xهای بزرگ‌تر همچنان ممکن است و به H2 برمی‌گردد.
- **allowlist فیلدها:** فیلدهای ساختگی (`lane_position`، `distance`، `robot` و ...) رد می‌شوند.
- **pickle:** `SandboxedReward` با `__getstate__`/`__setstate__` از سورس دوباره کامپایل می‌شود و `memory` را با خود نمی‌برد. درست است.
- **سرعت:** اعتبارسنجی هر کاندید حدود ۰٫۲۳ ثانیه است و گلوگاه نیست.
- **`lambda` با مقدار پیش‌فرض:** پذیرفته می‌شود و بی‌خطر است.

</div>

---

<div dir="rtl">

## ۵. جدول اولویت

| شناسه | سطح | خلاصه | هزینه‌ی اصلاح |
|---|---|---|---|
| H1 | بالا | `math` تزریق می‌شود ولی هر `math.*` رد می‌شود؛ پیام گمراه‌کننده | کم |
| H2 | بالا | خطای زمان اجرا در Stage III کل pipeline را قفل می‌کند | کم |
| M1 | متوسط | کندی وابسته به داده شناسایی نمی‌شود (۱٫۵ ساعت برای یک کاندید) | کم |
| M2 | متوسط | لیست/dict سراسری بین اپیزودها نشت می‌کند و پاداش غیرایستا می‌شود | کم |
| M3 | متوسط | بدون سقف برای بزرگی پاداش | کم |
| L1 | پایین | دسته‌بندی دلیل رد ناقص است | کم |
| L2 | پایین | `_SCALAR_FIELD_ATTRS` با نام‌های CrowdNav | کم |
| L3 | پایین | مسیر fallback مرده در `smoke_test_compute` | کم |
| L4 | پایین | `proc.close`، ترتیب `join`/`get`، حافظه‌ی مشترک در smoke | کم |

**ترتیب پیشنهادی اصلاح:** H2، H1، M2، M1، M3، و بعد Lها. H2 اول است چون تنها موردی است که می‌تواند کل اجرا را از کار بیندازد.

</div>

---

<div dir="rtl">

## ۶. وضعیت اصلاحات

همه‌ی یافته‌ها به همان ترتیب پیشنهادی اصلاح شدند. هر اصلاح یک commit جدا روی شاخه‌ی `fix/review-findings` دارد. حین کار یک یافته‌ی جدید هم پیدا شد (L5).

| شناسه | وضعیت | commit | چه تغییر کرد |
|---|---|---|---|
| H2 | اصلاح شد | `b1a1e26` | حلقه‌ی Stage III خطای `train_and_eval` را می‌گیرد. کاندید با `status="failed"` و متن خطا در `history` ثبت می‌شود، وارد `trained` و انتخاب بهترین نمی‌شود، checkpoint جلو می‌رود و اجرا ادامه پیدا می‌کند. resume وضعیت failed را بازیابی می‌کند. وقفه‌های سطح process (مثل Ctrl+C) همچنان بالا می‌روند. |
| H1 | اصلاح شد | `d2dc7a6` | `SAFE_MATH_ATTRIBUTES` در `config.py`: ‏`sqrt, exp, log, log1p, log10, tanh, sin, cos, tan, atan, atan2, hypot, fabs, floor, ceil, copysign, isfinite, isnan` و ثابت‌های `pi, e, tau, inf`. هر `math.<x>` دیگری با پیام «math.x is not allowed (allowed: ...)» رد می‌شود. پیام خطای import و هر سه پرامپت از همین ثابت ساخته می‌شوند. |
| M2 | اصلاح شد | `dad0c26` | `List`، `Dict` و `Set` در سطح ماژول (حتی داخل tuple) با پیام «mutable top-level values ... use the memory dict» رد می‌شوند. تست قبلی که `W = {...}` را مجاز می‌دانست به لیست ردشده‌ها منتقل شد. یک خط هم به پرامپت‌ها اضافه شد. |
| M1 | اصلاح شد | `d766311` | `SandboxedReward.compute` زمان هر فراخوانی را می‌سنجد: سقف تک‌فراخوانی ۱ ثانیه و سقف میانگین متحرک ۱۰۰ فراخوانی اخیر ۲۰ میلی‌ثانیه. smoke هم هر فراخوانی را با سقف ۰٫۱ ثانیه می‌سنجد. همه‌ی سقف‌ها در `SandboxConfig` قابل تنظیم‌اند. سقف تک‌فراخوانی عمداً شل است، چون Stage II در thread اجرا می‌شود و زمان انتظار GIL هم حساب می‌شود. |
| M3 | اصلاح شد | `8f5dc36` | `max_abs_reward = 1e4` در `SandboxConfig`. هم در smoke و هم در هر step آموزش، اگر `|total|` بیشتر شد، `RewardSandboxError` با پیام «reward magnitude too large». پرامپت‌ها عدد سقف را از همان config می‌خوانند. |
| L1 | اصلاح شد | `88585ee` | دسته‌های جدید: `not_on_allowlist`، `forbidden_construct`، `mutable_global`، `too_slow`، `reward_too_large`، `top_level_code`. timeout فقط با عبارت دقیق `execution exceeded timeout of` تشخیص داده می‌شود. تست‌ها دسته‌ها را از پیام‌های واقعی validator می‌سازند، پس اگر متن پیامی عوض شود تست می‌شکند. |
| L2 | اصلاح شد | `794cab0` | `_SCALAR_FIELD_ATTRS` حالا فیلدهای اسکالر highway است (`x, y, vx, vy, heading, speed, lane_index, on_road, progress, collision, off_road, timeout, time_step, global_time, time_limit`). `action` عمداً در آن نیست. |
| L3 + L4 | اصلاح شد | `6c001ee` | `smoke_test_compute(source_code, states, config)`: فقط یک مسیر (فرزند spawn شده) باقی ماند. `run_with_timeout` و مسیر fallback حذف شدند. نتیجه **قبل** از `join` خوانده می‌شود، فرزند در `finally` متوقف می‌شود و `queue.close()` و `proc.close()` صدا زده می‌شوند. اگر فرزند بدون نتیجه بمیرد، خطا فوراً گزارش می‌شود و تا پایان timeout صبر نمی‌شود. حافظه‌ی smoke هر بار که `global_time` جلو نرود پاک می‌شود (معادل `reset()` در مرز اپیزود). |
| L5 (جدید) | اصلاح شد | `1e9e583` | `make_highway_validator()` بدون آرگومان، `SandboxConfig()` معمولی و **بدون allowlist** می‌ساخت. pipeline اصلی مشکلی نداشت، چون pack خودش `highway_sandbox_config()` را می‌دهد. ولی تست‌ها و اسکریپت `collect_highway_stage1_dataset.py` بدون allowlist اجرا می‌شدند و مثلاً `state.lane_position` را می‌پذیرفتند. حالا مقدار پیش‌فرض `highway_sandbox_config()` است. |

**شواهد بعد از اصلاح** (همان اسکریپت‌های آزمایشی بخش ۲، دوباره اجرا شدند):

</div>

```text
math.exp / tanh / sqrt / log / pi        ACCEPT     (before: REJECT "hallucinated field?")
escape attempts (dunder, type, getattr,  REJECT     (unchanged)
  __builtins__, import, while, ...)
slow loop ~ ego.x  (range(x * 2000))     REJECT at smoke: one call took 0.164s (limit 0.1s)
1e15 * speed                             REJECT: reward magnitude too large
HIST = [] + HIST.append                  REJECT: mutable top-level values ...
dt = global_time - memory["t"]; 1/dt     ACCEPT     (before: false REJECT, ZeroDivisionError,
                                                     two dataset frames share global_time 20.2)
return 0.0                               ACCEPT     (left to Score1 "no_signal", by design)
validation latency per candidate         0.21 s     (before: 0.23 s)
```

<div dir="rtl">

**درباره‌ی ریسک حافظه‌ی مشترک در smoke (L4):** این ریسک با آزمایش تأیید شد. دو frame از دو trajectory مختلف در داده‌ی Stage I هر دو `global_time = 20.2` دارند. پس پاداشی که بر `dt` تقسیم می‌کند، در smoke با `ZeroDivisionError` رد می‌شد، در حالی که در محیط واقعی `dt` همیشه ۰٫۲ است. بعد از اصلاح پذیرفته می‌شود.

**تست‌ها:** ۲۳۲ تست سریع و تست کند PPO همه پاس می‌شوند. تست‌های جدید:

- `domains/highway/tests/test_sandbox_math.py`
- `domains/highway/tests/test_sandbox_runtime_limits.py`
- `domains/highway/tests/test_rejection_categories.py`
- تست‌های اضافه‌شده به `test_stage3_resume.py`، `test_sandbox_hardening_highway.py`، `test_sandbox_hardening.py` و `test_sandbox_timeout_kill.py`

**موارد باز (بخش‌های دیگر):**

- کاندیدی که Score1 آن را رد کرده باز هم با PPO آموزش داده می‌شود (`_needs_training`). در بخش ۷ بررسی می‌شود.

</div>
