# Audit فنی پایپ‌لاین RAISE (حوزهٔ فعال: Highway)

**مخاطب:** هر کسی که باید بفهمد سیستم از کجا شروع می‌شود، هر قطعه چه می‌کند، و داده چطور بین مراحل جریان دارد.  
**ریشهٔ کد:** `raise_env/`  
**مسیر فعال پایان‌نامه:** `--domain highway` (closed-loop)  
**مسیر قفل‌شده:** CrowdNav / Algorithm 1 baseline — فقط برای parity؛ در این سند به‌عنوان پس‌زمینه ذکر می‌شود، نه مسیر کار جاری.

تاریخ نگاشت: هم‌تراز با پروفایل‌های `highway_4h` / `highway_7h` و کالیبراسیون پیش‌فرض `no_speed_floor`.

---

## فهرست

1. [تصویر کلی](#1-تصویر-کلی)
2. [نقطه‌های ورود (CLI)](#2-نقطه‌های-ورود-cli)
3. [پروفایل‌ها و پیش‌فرض‌ها](#3-پروفایل‌ها-و-پیش‌فرض‌ها)
4. [Domain Pack](#4-domain-pack)
5. [محیط Highway و اکشن](#5-محیط-highway-و-اکشن)
6. [Stage I — داده و Score1](#6-stage-i--داده-و-score1)
7. [Sandbox پاداش](#7-sandbox-پاداش)
8. [لایهٔ LLM](#8-لایهٔ-llm)
9. [Evolver (تولید و تکامل ژنوم پاداش)](#9-evolver-تولید-و-تکامل-ژنوم-پاداش)
10. [حلقهٔ Closed-Loop](#10-حلقهٔ-closed-loop)
11. [Stage II — برچسب‌زنی با PPO کوتاه](#11-stage-ii--برچسب‌زنی-با-ppo-کوتاه)
12. [متریک‌ها، Fitness و Pareto](#12-متریک‌ها-fitness-و-pareto)
13. [Surrogate و Active Learning](#13-surrogate-و-active-learning)
14. [Proxy feedback و D.3 درون‌حلقه](#14-proxy-feedback-و-d3-درون‌حلقه)
15. [Stage III — اعتبارسنجی نهایی](#15-stage-iii--اعتبارسنجی-نهایی)
16. [آرتیفکت‌های خروجی](#16-آرتیفکت‌های-خروجی)
17. [جریان یک epoch به‌صورت قدم‌به‌قدم](#17-جریان-یک-epoch-به‌صورت-قدم‌به‌قدم)
18. [رفتار مشاهده‌شده و محدودیت‌های شناخته‌شده](#18-رفتار-مشاهده‌شده-و-محدودیت‌های-شناخته‌شده)
19. [نقشهٔ فایل‌های کلیدی](#19-نقشهٔ-فایل‌های-کلیدی)

---

## 1. تصویر کلی

هدف سیستم: **جستجوی برنامه‌ایِ تابع پاداش** (`compute_reward(state, memory)`) با LLM، فیلتر ارزان تحلیلی (Score1)، آموزش کوتاه RL (Stage II / PPO)، مدل جایگزین (surrogate)، انتخاب چندهدفه (Pareto)، و در پایان آموزش بلندتر (Stage III).

```text
جمع‌آوری trajهای Stage I (یک‌بار)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Closed-Loop (نوآوری / مسیر highway)                        │
│                                                           │
│  برای هر epoch g = 0 .. G-1:                              │
│    1) Score1 روی جمعیت پاداش‌ها                             │
│    2) Gate + AL → کدام کاندیدها Stage II بگیرند            │
│    3) PPO کوتاه (K2) روی holdout → متریک واقعی             │
│    4) Proxy evidence به LLM + اختیاری refine D.3           │
│    5) Refit surrogate (طبق بودجه برچسب)                    │
│    6) رتبهٔ Pareto / elite → نسل بعد (۲ crossover /        │
│       ۴ mutation / ۲ random برای N=8)                     │
└───────────────────────────────────────────────────────────┘
        │
        ▼
Stage III: PPO بلندتر (K3) روی نخبه‌ها → بهترین پاداش نهایی
```

**تفاوت با مسیر کلاسیک مقاله (CrowdNav):** در مسیر کلاسیک، Stage I کامل → سپس Stage II کامل روی همه → سپس Stage III. در closed-loop، Stage I و برچسب Stage II **درهم** اجرا می‌شوند تا هزینهٔ PPO فقط روی کاندیدهای انتخاب‌شده صرف شود.

---

## 2. نقطه‌های ورود (CLI)

| اسکریپت | نقش |
|---------|-----|
| `scripts/run_raise.py` | CLI عمومی → `RaisePipeline`. با `--profile` فلگ‌ها از `presets` تزریق می‌شوند. |
| `scripts/run_raise_highway_4h.py` | اورکستراسیون thesis: اطمینان از دیتاست Stage I، warm اختیاری، ساخت `output_dir`، فراخوانی `run_raise.py`، راهنمای resume. |
| `scripts/run_raise_highway_7h.py` | همان ۴h با `--profile highway_7h` (N=8, G=7). |
| `scripts/collect_highway_stage1_dataset.py` | ساخت دیتاست Score1. |
| `scripts/bootstrap_surrogate.py` | برچسب‌زنی اولیه برای surrogate گرم (اختیاری؛ پروفایل ۴h/۷h پیش‌فرض warm خالی است). |

دستور نوعی:

```powershell
cd raise_env
python scripts/run_raise_highway_7h.py --llm groq --skip-collect
```

Resume:

```powershell
python scripts/run_raise_highway_7h.py --llm groq --skip-collect --resume results/highway_7h_YYYYMMDD_HHMMSS
```

---

## 3. پروفایل‌ها و پیش‌فرض‌ها

منبع حقیقت: `raise_core/presets.py` → `CLOSED_LOOP_PROFILES`.

| پارامتر | `highway_4h` | `highway_7h` |
|---------|--------------|--------------|
| جمعیت N | 6 | **8** |
| نسل / epoch G | 4 | **7** |
| K2 (Stage II) | 12 000 **env_steps** | همان |
| K3 (Stage III) | 70 000 env_steps | همان |
| Stage2 eval | 20 اپیزود holdout | همان |
| Stage3 eval | 30 | همان |
| min_labels_gate | 16 | همان |
| AL max / epoch | 2 | همان |
| min_stage2 | 3 | همان |
| refit_every | 6 برچسب جدید | همان |
| evolve_rank | `pareto` | همان |
| elitism | True | همان |
| calibration | `no_speed_floor` | همان |
| eval_mode | `holdout_only` | همان |
| action_mode | `meta_default` (۱۱ دنده) | همان |
| warm_start PPO | True | همان |
| Pareto: progress / lane / overtake | همه True | همان |
| خروجی | `results/highway_4h_*` | `results/highway_7h_*` |

**منطق مقیاس زمان:** با ثابت ماندن K2/K3، دیوار ساعت تقریباً با `N×G` (تعداد برچسب‌های Stage II در حلقه) مقیاس می‌شود. ۷h عمداً فقط عمق جستجو را زیاد می‌کند، نه بودجهٔ هر PPO.

---

## 4. Domain Pack

الگوریتم هسته (`raise_core`) نباید به شبیه‌ساز خاص قفل باشد. هر محیط یک **pack** است:

| قطعه | Highway |
|------|---------|
| ثبت | `raise_core/domains/__init__.py` → `domains.highway` |
| ورود | `domains/highway/pack.py` → `get_pack()` |
| Prompt / seed | `prompts.py` |
| Score1 | `stage1.py` |
| Trainer II/III | `adapter.py` → SB3 PPO |
| Validator | `reward_checks.py` → `HighwayRewardValidator` |
| Spec انسانی | `spec.md` |

`RaisePipeline` با `--domain highway` این pack را لود می‌کند و Score1 / trainer / validator / promptها را از آن می‌گیرد. مسیر CrowdNav همچنان در رجیستری هست ولی برای کار thesis **منجمد** است.

---

## 5. محیط Highway و اکشن

### 5.1 محیط

- شناسه: `highway-fast-v0` (`env_wrapper.py`)
- مدت اپیزود: **۴۰ s**، فرکانس سیاست: **۵ Hz** → حدود **۲۰۰ گام**
- خطوط: ۴؛ مشاهده: kinematics (ego + حداکثر ۵ خودروی دیگر)
- پاداش‌های native محیط صفر شده‌اند؛ فقط پاداش تزریقی LLM / seed حاکم است
- terminate با off-road فعال است

### 5.2 دو رژیم ترافیک

| رژیم | `vehicles_count` | کاربرد |
|------|------------------|--------|
| Training | 20 | آموزش PPO |
| Holdout | **32** | ارزیابی رسمی انتخاب / fitness / Pareto |

Holdout عمداً شلوغ‌تر است تا «ثابت‌ماندن در لاین با سرعت ثابت» آسان نباشد. باند seed جدا (`HOLDOUT_SEED_OFFSET`) تا اپیزودهای ارزیابی با train هم‌پوشانی نداشته باشند.

پروفایل‌ها `highway_eval_mode=holdout_only` دارند: متریک انتخاب فقط از holdout می‌آید.

### 5.3 فضای اکشن (`action_config.py`)

سه حالت:

| حالت | نوع | شرح |
|------|-----|-----|
| **`meta_default`** (فعال) | `DiscreteMetaAction` | ۱۱ دنده `linspace(20,30,11)` = ۲۰٫۰، ۲۱٫۰، …، ۳۰٫۰؛ اکشن `Discrete(5)` |
| `meta_fine` | DiscreteMetaAction | دنده‌های ریزتر (مثلاً ۲۱ تایی در بازهٔ قابل تنظیم) |
| `continuous` | `ContinuousAction` | شتاب (±۵) و اختیاری فرمان |

ایندکس‌های Discrete(۵):

| ایندکس | نام | اثر |
|--------|-----|-----|
| 0 | LANE_LEFT | تغییر لاین چپ |
| 1 | IDLE | نگه‌داشتن هدف سرعت فعلی |
| 2 | LANE_RIGHT | تغییر لاین راست |
| 3 | FASTER | یک پله دنده بالاتر (نسبت به سرعت فیزیکی) |
| 4 | SLOWER | یک پله پایین‌تر |

**نکتهٔ مکانیکی مهم:** در highway-env، FASTER/SLOWER معمولاً بر اساس `speed_to_index(speed)` فعلی کار می‌کنند، نه لزوماً فقط ایندکس هدف ذخیره‌شده. بالا رفتن از ۲۰ به ۳۰ نیاز به چند FASTER پشت‌سرهم و زنده ماندن دارد؛ پایین آمدن تا کف ۲۰ با اسپم SLOWER مسیر کوتاه و امن‌تری است.

اسپان ego معمولاً نزدیک ~۲۵ m/s است؛ در holdout شلوغ این نقطه خطرناک‌تر از کف دنده است.

### 5.4 State قرارداد پاداش

`HighwayRewardState` فیلدهای مجاز برای LLM را تعریف می‌کند (سرعت، progress، collision، off_road، همسایه‌ها، …). فیلدهای CrowdNav (`robot`, `humans`, …) و `getattr` / `import` در sandbox رد می‌شوند.

---

## 6. Stage I — داده و Score1

### 6.1 جمع‌آوری دیتاست

`collect_highway_stage1_dataset.py` چند «رفتار» scripted را روی `meta_default` اجرا می‌کند و traj ذخیره می‌کند:

- موفقیت سریع (`safe_fast`) با rescale سرعت به باند ~[۲۲، ۲۵]
- موفقیت خزنده (`crawl`) → ~[۱۰، ۱۴] (منفی سخت برای Score1)
- lag / decoy_lag → ~[۱۵، ۱۷٫۵]
- collision / decoy_crash / random / swerve / idle
- در صورت کمبود timeout، نمونه‌های مصنوعی از donor موفقیت

خروجی: `domains/highway/data/stage1_dataset/{trajectories.jsonl,manifest.json}`.

**محدودیت:** این دیتاست همیشه meta است؛ با PPO continuous هم‌خوان نیست.

### 6.2 Score1 ترکیبی (`stage1.py`)

برای هر کاندید پاداش روی کل دیتاست:

\[
\begin{aligned}
\text{raw} &= 0.25\,\rho_{\text{Spearman}}(\text{rule}, \text{cum\_reward}) \\
&\quad + 0.40\,\text{AUC}_{\text{pref}}(\text{success} \succ \text{timeout} \succ \text{collision}) \\
&\quad + 0.35\,\text{throughput alignment} \\
&\quad - 0.50\,\text{crawl/lag penalty} \\
&\quad - 0.35\,\text{collision-decoy penalty}
\end{aligned}
\]

سپس clip حدودی به \([-1.5, 1.5]\).  
رد سخت فقط وقتی تقریباً هیچ سیگنالی نماند (تبهگنی شدید Spearman و pref/throughput≈۰).

**منطق:** Stage I باید پاداش‌های بی‌ربط / هک‌کنندهٔ crawl یا crash-bait را ارزان دور بریزد؛ جایگزین Stage II نیست.

---

## 7. Sandbox پاداش

مسیر: نرمال‌سازی نام تابع → `ast.parse` → سیاست ساختار/نام‌های ممنوع → اینترفیس `compute_reward(state, memory)` → exec محدود → smoke test روی stateهای ساختگی.

Highway اضافه می‌کند:

- چک literalهای `target_speed` خارج از بازهٔ دندهٔ قابل‌دسترس → `RewardSandboxError`
- هشدار برای literalهای سرعت در مقایسه‌ها

کاندید نامعتبر در evolver دوباره تولید می‌شود (تا سقف `max_invalid_replacements`)؛ در صورت اتمام بودجه، fallback به seed یا clone والد.

خطاهای نرم LLM (جواب خالی / truncate) نباید کل ران را بکشند؛ attempt می‌سوزد و در نهایت seed pad می‌شود.

---

## 8. لایهٔ LLM

| جزء | نقش |
|-----|-----|
| `raise_core/llm.py` | کلاینت‌ها: Groq، vLLM، Ollama، seed، scripted؛ استخراج بلوک ```python``` |
| `raise_core/key_manager.py` | استخر کلید `groq_keys.json`؛ چرخش روی 429؛ retry روی timeout / disconnect / **empty completion**؛ pacing بین درخواست‌ها |
| Promptها | `domains/highway/prompts.py` — D.1 اولیه/بچ، D.2 mutation/crossover، D.3 refine، D.4 دانش دامنه، D.5 seed |

پیش‌فرض مدل Groq در کد: خانوادهٔ gpt-oss روی Groq (قابل override با `--llm-model`).

تولید اولیه معمولاً **یک تماس بچ** برای N تابع است؛ اسلات‌های invalid تکی regenerate می‌شوند.

---

## 9. Evolver (تولید و تکامل ژنوم پاداش)

`raise_core/explore.py` — `StageIEvolver`

### 9.1 Gen0

1. یک prompt بچ D.1 → N سورس تابع  
2. validate هر کدام  
3. برای کمبود: regenerate تکی یا seed fallback  

### 9.2 ترکیب نسل بعد (برای N=۸ کاغذی)

| سطل | تعداد | منطق |
|-----|-------|------|
| Crossover | 2 | از top-2 رتبه |
| Mutation | 4 | روی نیمهٔ پایین‌تر + weakness / proxy evidence |
| Random | 2 | restart شبیه D.1 |

اگر `keep_runtime_elite` / elitism روشن باشد، یک اسلات از random (سپس mut/xover) کم می‌شود و elite جاری حفظ می‌شود.

برای N=۶ پروفایل ۴h: تخصیص به صورت `min(2,N)` crossover، سپس mutation، باقیمانده random (ممکن است random=۰ شود).

### 9.3 رتبه داخل evolver کلاسیک

در مسیر غیر closed-loop، رتبه عمدتاً Score1 است.  
در closed-loop، بعد از برچسب Stage II، **رتبهٔ پرورش** از `evolve_rank` (Pareto) می‌آید نه صرفاً Score1.

---

## 10. حلقهٔ Closed-Loop

هسته: `raise_core/raise_loop/runner.py` → `ClosedLoopRunner.run`

پیکربندی: `ClosedLoopConfig` (از `RaiseRunConfig` / پروفایل پر می‌شود).

مسیرها زیر `output_dir` ایزوله می‌شوند:

- `surrogate_model/`
- `surrogate_dataset/`
- `active_learning/`
- `closed_loop/` (checkpoint، epochs، pop snapshots، stage2_train، …)

### 10.1 Checkpoint / Resume

`closed_loop/checkpoint.json` (+ آینه در ریشهٔ run). Resume جمعیت، شمارهٔ epoch، مسیر surrogate و شمارندهٔ برچسب را برمی‌گرداند.

---

## 11. Stage II — برچسب‌زنی با PPO کوتاه

`domains/highway/adapter.py` — `HighwayPPOTrainer.train_and_eval`

1. Vec env آموزش با `training_env_config()` (۲۰ خودرو)  
2. PPO MlpPolicy، بودجهٔ **K2 env steps** (۱۲k در پروفایل)  
3. اختیاری **warm-start** از `model.zip` والد (اگر فضا سازگار باشد)  
4. ارزیابی روی holdout (۳۲ خودرو، ۲۰ اپیزود)  
5. نوشتن متریک‌ها روی `candidate.metadata["last_metrics"]` و append به surrogate dataset  

زمان تقریبی هر برچسب روی CPU: حدود ۳–۴ دقیقه train + ۱–۲ دقیقه eval (بسته به SR).

خروجی مدل: `closed_loop/stage2_train/r{round}_{candidate_id}/model.zip`

---

## 12. متریک‌ها، Fitness و Pareto

### 12.1 متریک‌های ارزیابی (`adapter` / `metrics`)

| متریک | معنی |
|-------|------|
| SR | نرخ اپیزود موفق (بدون collision/off-road تا پایان) |
| CR | نرخ برخورد |
| TR | نرخ off-road / timeout کلاس‌بندی‌شده |
| mean_speed / speed_p10 | سرعت ego |
| mean_progress | پیشرفت طولی تجمعی (متر) |
| soft_success | زنده **و** mean_speed ≥ ۲۵ **و** progress ≥ ۴۰۰ — **فقط تشخیصی** |
| lane_change_rate | تغییر لاین / گام |
| overtakes_per_km | سبقت‌های ثبت‌شده توسط `OvertakeTracker` نرمال‌شده با کیلومتر |

### 12.2 `highway_fitness` (اسکالر تشخیصی)

فرمول تقریبی: گیت سیگموید روی سرعت مؤثر × (SR−CR−۰٫۵·TR) + ترم‌های progress/speed/soft − جریمه‌های lag/تبهگنی.

**مهم:** در پروفایل فعلی، **پرورش جمعیت با این اسکالر نیست**؛ با Pareto است. Fitness برای لاگ، best_ever انسانی، و حالت‌های legacy/`population` می‌ماند.

### 12.3 Pareto (`pareto_rank.py`)

**کالیبراسیون پیش‌فرض `no_speed_floor`:**

- **Feasible** ⇔ `SR > 0` (بدون کف سرعت اجباری)
- اهداف بیشینه‌سازی (با علامت‌گذاری مناسب):  
  **`−CR`, `−TR`, `progress`, `mean_speed`, `lane_change_rate`, `overtakes_per_km`**
- SR و soft_success در بردار هدف نیستند
- غلبه با حاشیهٔ آماری تقریبی `2×Bernoulli SE` وقتی `n_eval` کافی باشد
- رتبه‌بندی: ابتدا feasibleها با جبهه‌های غیرمغلوب + crowding؛ سپس infeasibleها

**حالت‌های دیگر (میراث):**

- `population`: مرجع از صدک‌ها / endpoints دندهٔ SLOWER–FASTER + سقف CR/TR  
- `env_measured`: بقا مثل no_speed_floor + آمار ambient فقط اطلاع‌رسانی  

**Elite archive (`auto`):** در `no_speed_floor` → حالت Pareto؛ elite برای elitism نسل بعد استفاده می‌شود.

---

## 13. Surrogate و Active Learning

### 13.1 Surrogate

مدل bagged رگرسیون روی ویژگی‌های ارزان کد/Score1 → پیش‌بینی اهداف چندتایی (برای highway: SR, CR, TR, mean_speed, mean_progress, soft_success).

- Fit اولیه: بعد از Gen0 اگر حداقل یک برچسب باشد  
- Refit: هر `refit_every` برچسب جدید (۶ در پروفایل)  
- کیفیت: MAE اعتبارسنجی؛ آستانهٔ highway برای «آماده بودن گیت سخت»: `max_val_mae_gate=0.35` روی اهداف نرخ  

### 13.2 Gate نرم در برابر سخت (`epoch.select_to_label`)

| شرط | رفتار |
|-----|--------|
| epoch 0 یا مدل نارس یا برچسب < `min_labels_gate` و MAE ناکافی | **Soft:** تقریباً همه برای Stage II انتخاب می‌شوند |
| مدل آماده | **Hard:** بخشی از کاندیدهای «ضعیف + مطمئن» drop می‌شوند؛ بقیه نگه داشته می‌شوند |

### 13.3 Active Learning

بعد از گیت سخت، تا `al_max` کوئری اضافه برای برچسب Stage II بر اساس عدم‌قطعیت / کیفیت پیش‌بینی surrogate. حداقل `min_stage2` برچسب Stage II در epoch حفظ می‌شود.

---

## 14. Proxy feedback و D.3 درون‌حلقه

بعد از برچسب واقعی:

1. بلوک **evidence** (اعداد holdout + وضعیت Pareto/feasibility) به metadata کاندید می‌چسبد  
2. Mutation بعدی این اعداد را در weakness می‌بیند (بدون لیبل‌های اخلاقی ثابت مثل «hack»)  
3. اختیاری: بدترین کاندید(ها) با prompt D.3 بازنویسی می‌شوند (`proxy_d3_per_epoch`، از وقتی بودجهٔ برچسب اجازه دهد)

اگر LLM خراب شود، کاندید قبلی حفظ می‌شود (soft fail).

---

## 15. Stage III — اعتبارسنجی نهایی

پس از اتمام G epoch:

1. گیت surrogate روی جمعیت نهایی (در صورت آمادگی)  
2. مونتاژ مجموعهٔ Stage III (elites / بهترین‌ها بر اساس سیاست pack)  
3. `Stage3Runner`: PPO با **K3=70k**، eval=۳۰، معمولاً **بدون H-sweep** در پروفایل highway  
4. خروجی: `stage3_population.json`, `best_stage3.json`, مدل‌های `stage3_train/`

این مرحله ادعای نهایی کیفیت پاداش را با بودجهٔ بیشتر می‌سنجد؛ جایگزین حلقهٔ جستجو نیست.

---

## 16. آرتیفکت‌های خروجی

مثال: `results/highway_7h_YYYYMMDD_HHMMSS/`

```text
config.json                 # فلگ‌های مؤثر
seed_reward.py              # D5 کپی‌شده
checkpoint.json / RESUME.json
stage1_population.json
best_stage1.json
stage2_population.json
best_stage2.json
surrogate_model/            # model.joblib, metrics.json, …
surrogate_dataset/          # features.jsonl, labels.jsonl
active_learning/
closed_loop/
  checkpoint.json
  epochs.jsonl              # خلاصهٔ هر epoch
  pop_epoch_XXX.json        # رتبه / Pareto فشرده
  stage1_rejections.jsonl
  al_steps.jsonl
  stage2_train/r##_cid/     # model.zip + diagnostics
stage3_*/                   # بعد از اتمام
plots/                      # در صورت viz پس‌ازاجرا
```

---

## 17. جریان یک epoch به‌صورت قدم‌به‌قدم

برای epoch `g` با جمعیت فعلی `P`:

1. **Score1** روی همهٔ اعضای `P` (ارزان، دیتاست ثابت).  
2. **پیش‌بینی surrogate** (اگر مدل هست) → کیفیت/عدم‌قطعیت.  
3. **`select_to_label`:** لیست `to_label` + گزارش soft/hard + AL.  
4. برای هر id در `to_label`:  
   - PPO K2 روی train traffic  
   - eval holdout → متریک  
   - append به dataset surrogate  
5. **Proxy evidence** (+ حداکثر چند D.3).  
6. **Refit** اگر آستانهٔ برچسب جدید رسید.  
7. **Pareto stamp + rank_for_evolution** (+ diversity / elite).  
8. نوشتن `epochs.jsonl` و `pop_epoch_g.json`.  
9. اگر `g+1 < G`: `_next_generation(ranked)` → جمعیت epoch بعد.  
10. Checkpoint.

Gen0 معمولاً همه را برچسب می‌زند (گیت نرم) تا surrogate از صفر ساخته شود.

---

## 18. رفتار مشاهده‌شده و محدودیت‌های شناخته‌شده

این‌ها باگ «۱۱ دنده وصل نیست» نیستند؛ خواص سیستم فعلی‌اند:

1. **جاذبٔ خزیدن روی دندهٔ ۲۰** در برابر **تصادف در سرعت بالا** روی holdout ۳۲تایی؛ اسپان ~۲۵ خطرناک است و سیاست «همیشه SLOWER» آسان یاد گرفته می‌شود.  
2. **دنده‌های میانی (۲۱–۲۳) در شبیه‌ساز قابل بقا هستند** (حتی با IDLE)، ولی تحت PPO کوتاه + warm-start از elite خزنده + اهداف progress/−CR کمتر به‌عنوان سیاست پایدار ظاهر می‌شوند.  
3. **`soft_success` در Pareto نیست** → فشار صریح برای «زنده نزدیک ۲۵» در انتخاب پرورش ضعیف است.  
4. **`overtakes_per_km` روی crashها می‌تواند بالا برود** (شخم ترافیک) ولی feasibleها اغلب lane/overtake≈۰ دارند.  
5. **Score1 بالا روی crawlerهای موفق** با Stage II هم‌راستا نیست همیشه.  
6. **کالکتور Stage I همیشه meta است**؛ با `continuous` هم‌خوان نیست.  
7. **LLM/API flake** (empty, truncate, proxy) اگر soft-fail نباشد ران را می‌کشد؛ لایهٔ retry/soft-fail برای این طراحی شده است.

---

## 19. نقشهٔ فایل‌های کلیدی

| مسیر | مسئولیت |
|------|-----------|
| `scripts/run_raise.py` | CLI عمومی |
| `scripts/run_raise_highway_4h.py` / `_7h.py` | اورکستراسیون thesis |
| `raise_core/pipeline.py` | `RaisePipeline` |
| `raise_core/presets.py` | پروفایل‌های قفل‌شده |
| `raise_core/explore.py` | Evolver Stage I |
| `raise_core/llm.py` / `key_manager.py` | LLM و کلیدها |
| `raise_core/raise_loop/runner.py` | حلقهٔ closed-loop |
| `raise_core/raise_loop/epoch.py` | Gate + AL selection |
| `raise_core/raise_loop/evolve_rank.py` | اتصال Pareto / elite |
| `raise_core/raise_loop/proxy_feedback.py` | Evidence + D.3 |
| `raise_core/surrogate/*` | مدل، bootstrap، gate |
| `raise_core/validate.py` | Stage III runner |
| `raise_core/sandbox/*` | اعتبارسنجی کد پاداش |
| `domains/highway/pack.py` | مونتاژ pack |
| `domains/highway/env_wrapper.py` | Env و holdout/train |
| `domains/highway/action_config.py` | دنده‌ها / continuous |
| `domains/highway/adapter.py` | PPO train/eval |
| `domains/highway/metrics.py` | Fitness اسکالر |
| `domains/highway/pareto_rank.py` | انتخاب چندهدفه |
| `domains/highway/stage1.py` | Score1 |
| `domains/highway/prompts.py` | متن LLM + seed |
| `domains/highway/overtake.py` | شمارش سبقت |

---

## خلاصهٔ یک‌خطی برای خوانندهٔ عجول

**RAISE highway** یک حلقهٔ بسته است که با LLM تابع پاداش می‌سازد، با Score1 فیلتر می‌کند، با PPO کوتاه روی ترافیک شلوغ holdout برچسب واقعی می‌زند، با surrogate/AL هزینه را کم می‌کند، با Pareto چندهدفه والدین نسل بعد را انتخاب می‌کند، و در پایان با PPO بلندتر (K3) اعتبارسنجی می‌کند. پروفایل‌های `highway_4h` و `highway_7h` فقط عمق جستجو (N×G) را عوض می‌کنند؛ K2/K3 و منطق انتخاب یکسان‌اند.
