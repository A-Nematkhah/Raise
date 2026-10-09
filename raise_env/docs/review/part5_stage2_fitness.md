<div dir="rtl">

# بررسی بخش ۵: Stage II، آموزش PPO و معیار fitness

**دامنه:**

- `domains/highway/adapter.py` (~۶۷۰ خط)
- `domains/highway/metrics.py` (~۲۹۰ خط)
- `domains/highway/objective_constants.py` (~۲۶ خط)
- `domains/highway/watchdog.py` (~۱۳۴ خط)
- `raise_core/refine.py` (~۱۲۸ خط)

نقاط اتصال هم بررسی شد: `raise_loop/runner.py` (seed ارزیابی و warm start)، `parallel_label.attach_warm_start_checkpoints`، `selection.py`، `env_wrapper.holdout_env_config`، و پرامپت‌های fitness در `prompts.py`.

**روش:** کد خط به خط خوانده شد. فرمول fitness روی سناریوهای مصنوعی تجزیه شد. policy ثابت «همیشه SLOWER» روی holdout جدید با ۶ seed ارزیابی جداگانه (هر کدام ۲۰ اپیزود) اندازه‌گیری شد. اعداد بخش «شواهد» از همین اجراهاست. یافته‌ی H3 بخش ۱ اینجا با اندازه‌گیری تازه تکمیل شده است.

> بخش‌های ۱ تا ۵ این گزارش وضعیت کد **قبل از اصلاح** را توصیف می‌کنند. وضعیت اصلاحات و شواهد بعد از اصلاح در بخش ۶ آمده است.

**سطح‌بندی یافته‌ها:**

- **H (بالا):** روی اعتبار نتایج پایان‌نامه یا رتبه‌بندی کاندیدها اثر مستقیم دارد.
- **M (متوسط):** باگ یا سوگیری در مسیر واقعی، یا هزینه‌ی محاسباتی قابل‌توجه.
- **L (پایین):** تمیزکاری، ناسازگاری مستندات، یا کد مرده.

</div>

---

<div dir="rtl">

## ۱. خلاصه‌ی معماری

جریان یک برچسب Stage II برای هر کاندید:

</div>

```text
ClosedLoopRunner (per generation g)
  stage2_cfg.highway_eval_seed_base = seed + g*1000   # shared across candidates
  attach_warm_start_checkpoints(children ← parent.zip)
  for each candidate i:
      train seed = cfg.seed + g*1000 + i
      PPO(MlpPolicy, n_steps=64, batch=64, n_epochs=10, lr=3e-4)
          optional: load parent policy weights
          learn(12_000 env steps)   # ~60 episodes @ 200 steps
      eval holdout (20 eps, seed_base shared, stochastic by default)
      metrics_dict["holdout"] = extras + SR/CR/TR/...
      fitness = highway_fitness(holdout)   # selection objective
```

<div dir="rtl">

**فرمول رسمی** (`metrics.py`):

</div>

```text
v_eff = speed_p10 (else mean_speed)
if v_eff < V_FLOOR(3.75):  return -10          # hard DQ

gate = sigmoid(6 · (v_eff − 17.5) / 15)         # center of [V_MIN=10, V_TARGET=25]

F = gate · (SR − CR − 0.5·TR)
  + 0.35 · tanh(progress / 800)                 # NOT gated
  + 0.25 · tanh(mean_speed / 25)                # NOT gated
  + 0.15 · soft_success · gate
  − lag_penalty − degeneracy_penalty − low_speed_penalty
```

<div dir="rtl">

با `SR+CR+TR = 1` و `TR=0` داریم: `survival = 2·SR − 1`. پس در SR=۰٫۵۵، سهم بقا فقط ۰٫۱۰ است و بعد از gate (~۰٫۷۷) می‌شود ~۰٫۰۸ — در حالی که سقف ترم‌های پیشرفت+سرعت بدون gate برابر ۰٫۶۰ است.

**نکته‌ی مثبت:** داخل هر نسل، `highway_eval_seed_base` برای همه‌ی کاندیدها یکسان است (`runner.py` خط ۴۱۷). پس رتبه‌بندی نسبی داخل نسل عادلانه‌تر از مقایسه‌ی مطلق بین نسل‌هاست.

</div>

---

<div dir="rtl">

## ۲. یافته‌ها

### H1. ترم‌های progress و speed بدون gate/survival هستند — policy بدیهی «همیشه SLOWER» fitness مثبت و بالا می‌گیرد

**محل:** `metrics.py` خطوط ۱۸۸ تا ۲۰۶. منتقل‌شده از H3 بخش ۱.

**مشکل:** سهم بقا با gate ضرب می‌شود، ولی `0.35·tanh(progress/800)` و `0.25·tanh(speed/25)` همیشه اضافه می‌شوند — حتی وقتی SR پایین یا survival منفی است. در نتیجه یک policy که فقط `SLOWER` می‌زند (کند، بدون تغییر لاین، ~۴۵٪ تصادف روی holdout چگال) همچنان fitness مثبت می‌گیرد، چون پیشرفت و سرعتِ اپیزودهای زنده‌مانده ترم‌های ungated را پر می‌کنند.

**شواهد — تجزیه روی متریک‌های شبیه H3:**

</div>

```text
always_slower (SR=0.55 CR=0.45 speed=20.5 pl≈560):
  gate=0.769  surv=+0.10  gate*surv=+0.077
  prog≈+0.21  spd≈+0.17   ungated_bonus≈+0.38
  lag=degen=low=0
  → fitness ≈ 0.45   (real holdout seed=425)

perfect_safe_slow (SR=1 speed=20)     fitness ≈ 0.69  (degen fires)
dead_but_fast_progress (SR=0 CR=1)    fitness ≈ -0.50
  (gate*surv=-0.93 but ungated_bonus=+0.43 cushions the crash)
```

<div dir="rtl">

**اندازه‌گیری واقعی holdout (density=1.5)، policy ثابت SLOWER، ۲۰ اپیزود، ۶ seed:**

</div>

```text
seed  SR    CR    progress  speed   fitness
425   0.55  0.45  561 m     20.5    0.455
426   0.80  0.20  682 m     20.5    0.859
427   0.80  0.20  678 m     20.5    0.858
428   0.55  0.45  548 m     20.7    0.452
429   0.40  0.60  457 m     20.9    0.202
430   0.35  0.65  454 m     20.8    0.126

mean fitness = 0.492   stdev = 0.286   range = [0.126, 0.859]
```

<div dir="rtl">

**اثر روی پایان‌نامه:** بهترین Stage III گزارش‌شده‌ی قبلی (~۰٫۴۰۸) **زیر میانگین** همین policy ثابت و داخل یک انحراف معیار آن است. بدون baseline ثابت «همیشه SLOWER / همیشه IDLE»، نمی‌توان فهمید حلقه‌ی تکاملی از یک heuristic ساده بهتر شده یا نه. همچنین در بخش ۱ دیده شد که PPO با پاداش seed در ۱۲k step به همین رفتار SLOWER همگرا می‌شود — یعنی fitness فعلی این attractor را پاداش می‌دهد.

**پیشنهاد اصلاح (یکی یا ترکیبی):**

- **(الف) پیشنهادی:** `progress_term` و `speed_term` را هم در `gate` ضرب کن، یا فقط وقتی `survival > 0` (یعنی `SR > 0.5` در حالت TR=۰) فعال شوند.
- **(ب):** جریمه‌ی تصادف را سنگین‌تر کن، مثلاً `survival = SR − 2·CR − TR`، تا SR=۰٫۵۵ دیگر مثبت نباشد.
- **(ج):** یک کف سخت: اگر `SR < 0.7` یا `CR > 0.3` آنگاه fitness سقف داشته باشد یا DQ نرم بگیرد.
- بعد از تغییر، همین جدول always-SLOWER را دوباره اندازه بگیر؛ هدف معقول: fitness همین policy نزدیک صفر یا منفی، و یک policy خوبِ آموزش‌دیده به‌وضوح بالاتر.

</div>

---

<div dir="rtl">

### H2. نویز ارزیابی با ۲۰ اپیزود برای رتبه‌بندی و ادعای پایان‌نامه خیلی بزرگ است

**محل:** `refine.py` (`eval_episodes=20`)، `adapter._eval_metrics`، پیش‌فرض `highway_eval_deterministic=False`.

**مشکل:**

- برای یک policy **کاملاً ثابت و قطعی**، با عوض کردن فقط seed ارزیابی، fitness از **۰٫۱۳ تا ۰٫۸۶** جابه‌جا شد (stdev ≈ ۰٫۲۹).
- SE دوجمله‌ای نظری برای SR با n=۲۰ حدود ۰٫۱۱ است؛ در عمل stdev(SR)=۰٫۱۸ هم دیده شد (ترافیک holdout ناهمگن است).
- ارزیابی پیش‌فرض **stochastic** است (`Stage2Config.highway_eval_deterministic=False`). یعنی حتی با seed base مشترک، نمونه‌گیری عمل PPO نویز اضافه می‌کند. (در `adapter` مقدار پیش‌فرض `getattr(..., True)` است، ولی `Stage2Config` مقدار `False` دارد و pipeline از config استفاده می‌کند.)

**نکته‌ی مثبت:** داخل یک نسل، `highway_eval_seed_base` مشترک است؛ پس همه‌ی کاندیدها روی همان ۲۰ سناریو ارزیابی می‌شوند و رتبه‌بندی نسبی پایدارتر از مقایسه‌ی مطلق بین اجراهاست. ولی:

- مقایسه‌ی fitness بین نسل‌ها (چون seed base با `g*1000` عوض می‌شود) و با baselineهای خارجی پرنویز است.
- با ارزیابی stochastic، دو کاندید نزدیک هم می‌توانند فقط به‌خاطر نمونه‌ی عمل جابه‌جا شوند.

**پیشنهاد اصلاح:**

- ارزیابی را **deterministic** کن (`highway_eval_deterministic=True`) مگر برای یک گزارش جداگانه‌ی stochastic.
- تعداد اپیزود holdout را برای Stage II به ۳۰–۴۰ و برای Stage III (که الان ۳۰ است) در صورت نیاز بالاتر ببر؛ یا CI/bootstrap روی fitness گزارش کن.
- در گزارش پایان‌نامه، همیشه baseline ثابت (SLOWER/IDLE) را با **همان** `eval_seed_base` کنار کاندیدها بگذار.

</div>

---

<div dir="rtl">

### H3. `soft_success` در فرمول fitness هست، در حالی که مستندات می‌گویند فقط diagnostic است

**محل:** `metrics.py` خط ۱۹۲؛ `objective_constants.py` خط ۹–۱۰؛ `adapter._eval_metrics` خطوط ۱۴۷–۱۵۱؛ `prompts.py` (~خط ۲۲۸)؛ `pack.py` metadata.

**مشکل:**

- `objective_constants` می‌گوید soft_success فقط برای لاگ است و نباید در پرامپت بیاید.
- `adapter` می‌گوید «Human-facing diagnostic only: NOT a selection / LLM objective».
- ولی فرمول واقعی `+ 0.15 · soft_success · gate` دارد. اختلاف دو کاندید با متریک یکسان و soft برابر ۰ در برابر ۱ حدود **۰٫۱۵** در fitness است (در probe: ۰٫۹۶۹ → ۱٫۱۱۴).
- پرامپت به LLM می‌گوید soft «small fitness bonus» است ولی «برایش طراحی نکن» — یعنی عملاً در objective هست و مدل می‌تواند برای آستانه‌ی ۲۵ m/s و پیشرفت ۴۰۰ m بهینه‌سازی کند.

**پیشنهاد اصلاح:** یکی را انتخاب کن و همه‌جا یکسان بنویس:

- **(الف) پیشنهادی برای پایان‌نامه:** soft را از fitness حذف کن؛ فقط در لاگ/نمودار بماند.
- **(ب):** اگر می‌خواهی بماند، مستندات و `objective_constants` را اصلاح کن و آستانه‌ها را صریح در spec بیاور.

</div>

---

<div dir="rtl">

### M1. Warm start به فرزندان crossover/mutation مزیت ساختاری می‌دهد

**محل:** `parallel_label.attach_warm_start_checkpoints`؛ `adapter` کپی `policy.state_dict`؛ پیش‌فرض `highway_warm_start=True`؛ سهم عملگرها در `explore.py`: crossover=۲، mutation=۴، random=۲.

**مشکل:**

- ۶ از ۸ اسلات نسل بعد (در صورت داشتن `parent_ids` و checkpoint والد) با وزن‌های PPO والد شروع می‌کنند؛ ۲ اسلات random از صفر.
- وزن‌ها زیر **پاداش والد** یاد گرفته شده‌اند؛ value function برای پاداش فرزند اشتباه است. این می‌تواند کمک یا آسیب بزند، ولی در هر حال مقایسه‌ی عادلانه‌ی «کیفیت پاداش» را با «کیفیت initialization سیاست» قاطی می‌کند.
- Gen0 همه از صفرند؛ از Gen1 به بعد، فرزندان نسبت به immigrantهای تصادفی مزیت محاسباتی دارند (۱۲k step روی یک policy گرم در برابر ۱۲k از صفر).

**اثر:** ممکن است حلقه به خاطر warm start پیشرفت نشان دهد، نه لزوماً به‌خاطر بهتر بودن کد پاداش. برای ادعای علمی لازم است یا warm start خاموش شود، یا یک ablatio n گزارش شود.

**پیشنهاد اصلاح:**

- برای آزمایش اصلی پایان‌نامه: `highway_warm_start=False`، یا
- دو بازوی موازی (با/بدون warm start) با بودجه یکسان، یا
- فقط weights سیاست را منتقل کن و value head را از نو initialize کن (در SB3 جدا کردن کمی کار می‌برد).

</div>

---

<div dir="rtl">

### M2. بودجه‌ی PPO Stage II کوتاه است و هایپرپارامترها تهاجمی نیستند ولی با fitness فعلی گمراه‌کننده‌اند

**محل:** `Stage2Config`: `train_env_steps=12_000`، `highway_n_steps=64`، `n_epochs=10`، `ent_coef=0.0`.

**حساب:**

</div>

```text
updates ≈ 12000 / 64 ≈ 187
epoch-grads ≈ 1870
train episodes ≈ 60 (200 steps/ep)
batch_size = 64 (= whole rollout) → 1 minibatch per epoch
```

<div dir="rtl">

**مشکل:** ۶۰ اپیزود برای یادگیری تغییر لاین در ترافیک چگال کم است. با `ent_coef=0` اکتشاف عمدتاً از نویز اولیه‌ی Gaussian/Categorical است. در بخش ۱، پاداش seed در همین بودجه به always-SLOWER همگرا شد. افزایش بودجه alone بدون اصلاح H1 فقط همان attractor را با اطمینان بیشتر یاد می‌دهد.

**پیشنهاد:** بعد از اصلاح fitness، یک منحنی یادگیری (۴k / ۱۲k / ۲۴k / ۴۸k) با پاداش seed و یک پاداش خوب دستی رسم شود. فعلاً افزایش کورکورانه‌ی K2 توصیه نمی‌شود.

</div>

---

<div dir="rtl">

### M3. Detector انحطاط (degeneracy) روی always-SLOWER آتش نمی‌گیرد

**محل:** `metrics._degeneracy_penalty`.

**مشکل:** شرط‌ها اغلب `SR >= 0.9` یا `0.95` می‌خواهند. always-SLOWER با SR≈۰٫۵۵ از همه‌ی آن‌ها رد می‌شود (`pen=0`)، در حالی که دقیقاً همان exploit «کند بران و زنده بمان» است که fitness را باد می‌کند. وقتی SR اتفاقی ۰٫۸۰ شود (seedهای ۴۲۶/۴۲۷)، هنوز `SR < 0.9` است و lane_change≈۰ هم جریمه نمی‌گیرد.

**پیشنهاد:** آستانه‌ی SR را پایین بیاور (مثلاً ۰٫۵) و/یا جریمه‌ی «lane_change_rate < 0.02 و mean_speed < V_TARGET» را مستقل از SR بالا اضافه کن — یا بهتر، ریشه‌ی H1 را ببند تا به detector وابسته نباشی.

</div>

---

<div dir="rtl">

### L1. ناسازگاری پیش‌فرض `eval_deterministic` بین adapter و Stage2Config

**محل:** `adapter.py` خط ۵۶۶: `getattr(config, "highway_eval_deterministic", True)`؛ `refine.Stage2Config` و `validate.Stage3Config`: پیش‌فرض `False`.

اگر کسی config بدون این فیلد بدهد، رفتار قطعی می‌شود؛ مسیر pipeline stochastic است. گیج‌کننده است.

**پیشنهاد:** یک پیش‌فرض واحد (ترجیحاً `True` بعد از H2) در `Stage2Config` / `Stage3Config` و همان را در `getattr` بگذار.

</div>

---

<div dir="rtl">

### L2. آستانه‌ی progress برای soft_success (۴۰۰ m) جادویی و خارج از `objective_constants` است

**محل:** `adapter._eval_metrics(..., soft_progress_m=400.0)`.

با `V_TARGET=25` و اپیزود ۴۰ s، پیشرفت ایده‌آل ~۱۰۰۰ m است؛ ۴۰۰ m یعنی زنده ماندن با ~۱۰ m/s میانگین. این عدد در constants مرکزی نیست.

</div>

---

<div dir="rtl">

### L3. StallWatchdog فقط hang کامل را می‌گیرد

**محل:** `watchdog.py`، پیش‌فرض ۳۰۰ s بدون heartbeat.

با اصلاح M1 بخش ۲ (سقف زمان هر `compute`)، این لایه بیشتر برای فریزهای native/C باقی می‌ماند. محدودیت async exception روی فراخوانی‌های C همچنان هست و در docstring آمده — قابل قبول است.

</div>

---

<div dir="rtl">

### L4. CUDA عملاً خاموش است مگر `RAISE_HIGHWAY_FORCE_CUDA=1`

**محل:** `adapter._train_and_eval` خطوط ۴۴۸–۴۵۷.

برای MLP کوچک روی highway این معمولاً درست است (SB3 هم CPU را توصیه می‌کند). فقط اگر کسی `--device cuda` بدهد و انتظار سرعت داشته باشد، غافلگیر می‌شود. یک log واضح‌تر کافی است.

</div>

---

<div dir="rtl">

## ۳. مواردی که بررسی و تأیید شد

- **ترجیح holdout در fitness:** `_metrics_for_fitness` و تست `test_metrics_antihack` — وقتی کلید تو در تو `holdout` باشد، selection از آن استفاده می‌کند؛ `train_dist` فقط برای لاگ است.
- **DQ خزیدن:** `v_eff < 3.75` → fitness = −۱۰.
- **Gate در بازه‌ی [۱۰، ۲۵]:** در v=۱۰ gate≈۰٫۰۵، در v=۱۷٫۵ برابر ۰٫۵، در v=۲۵≈۰٫۹۵.
- **ایزوله بودن memory در VecEnv:** `clone_reward_fn` با deepcopy؛ برای `SandboxedReward` pickle/recompile درست است.
- **Warm start و action space:** اگر فضای عمل والد/فرزند ناسازگار باشد، warm start رد می‌شود.
- **Seed ارزیابی مشترک در نسل:** `highway_eval_seed_base = seed + g*1000`.
- **Monitor فقط روی train:** eval بدون Monitor — درست برای متریک‌های دستی.
- **Watchdog:** heartbeat روی هر env step؛ محدودیت C-call مستند شده.

</div>

---

<div dir="rtl">

## ۴. موارد خارج از دامنه

| مورد | بخش مقصد |
|---|---|
| سهم عملگرهای evolution و reflection | ۶ |
| `_needs_training` برای کاندیدهای `score1_rejected` | ۷ |
| بودجه‌ی Stage III و انتخاب برنده با ۳۰ اپیزود | ۸ |
| پرامپت‌هایی که جزئیات fitness را لو می‌دهند (reward hacking متنی) | ۳ |

</div>

---

<div dir="rtl">

## ۵. جدول اولویت

| شناسه | سطح | خلاصه | هزینه‌ی اصلاح |
|---|---|---|---|
| H1 | بالا | progress/speed بدون gate؛ always-SLOWER fitness مثبت (تا ۰٫۸۶) | کم–متوسط |
| H2 | بالا | نویز ۲۰ اپیزود: fitness همان policy در بازه‌ی ۰٫۱۳–۰٫۸۶؛ eval stochastic | کم |
| H3 | بالا | soft_success در fitness خلاف مستندات diagnostic | کم |
| M1 | متوسط | warm start به ۶/۸ فرزند مزیت می‌دهد | کم (flag) / متوسط (ablation) |
| M2 | متوسط | بودجه‌ی ۱۲k / ۶۰ اپیزود؛ بعد از H1 دوباره سنجش شود | سنجش |
| M3 | متوسط | degeneracy روی SLOWER آتش نمی‌گیرد | کم |
| L1–L4 | پایین | پیش‌فرض deterministic، آستانه‌ی soft، لاگ CUDA، watchdog | کم |

**ترتیب پیشنهادی اصلاح:** H1 → H3 → H2 → M3 → M1، سپس سنجش دوباره‌ی M2.

معیار موفقیت پیشنهادی برای H1+H2: fitness میانگین always-SLOWER روی ۶ seed ارزیابی ≤ ۰٫۰۵ (یا منفی)، و stdev با eval قطعی و n≥۳۰ به‌طور محسوس کمتر از ۰٫۲۹ فعلی.

</div>

---

<div dir="rtl">

## ۶. وضعیت اصلاحات

همه‌ی یافته‌ها به ترتیب پیشنهادی (H1 → M3 → H3 → H2 → M1، سپس L) اصلاح شدند و هر کدام یک commit جدا روی شاخه‌ی `fix/review-findings` دارد. در یک مورد عمداً از پیشنهاد بخش ۲ فاصله گرفتم: **ارزیابی stochastic ماند** (توضیح در ردیف H2).

| شناسه | وضعیت | commit | چه تغییر کرد |
|---|---|---|---|
| H1 | اصلاح شد | `aabe1a6` | `survival = SR − 2·CR − TR` (ثابت‌های `CRASH_WEIGHT=2` و `OFF_ROAD_WEIGHT=1` در `objective_constants`). ترم‌های پیشرفت و سرعت حالا در `max(0, survival)` ضرب می‌شوند؛ یعنی مسافتی که قبل از تصادف رانده شده دیگر تصادف را جبران نمی‌کند. `fitness_components` مقدار `survival` را هم گزارش می‌کند. متن هدف در پرامپت (`_SELECTION_OBJECTIVE`) و `spec.md` با فرمول جدید یکی شد. |
| M3 | اصلاح شد | `64ff53c` | جریمه‌ی «بدون تغییر لاین» (۰٫۱۵) قبلاً فقط با `SR ≥ 0.9` فعال می‌شد. حالا وقتی `lane_change_rate < 0.02` و `mean_speed < V_TARGET` باشد، **مستقل از SR** هم فعال می‌شود. بقیه‌ی بررسی‌های degeneracy همان شرط `SR ≥ 0.5` را دارند. |
| H3 | اصلاح شد | `0c67cf6` | ترم `0.15·soft_success·gate` از fitness حذف شد. `soft_success` فقط در متریک‌ها و لاگ می‌ماند. خط مربوط به آن از پرامپت D4 هم حذف شد. تست جدید: تغییر `soft_success` از ۰ به ۱ هیچ اثری روی fitness ندارد. |
| H2 | اصلاح شد (با تغییر رویکرد) | `1ce9773` | تعداد اپیزود ارزیابی Stage II از ۲۰ به **۳۰** رسید (`Stage2Config`، `RaiseRunConfig`، fallback در `runner`). هر ارزیابی حالا `fitness_se` هم دارد: انحراف معیار fitness روی ۲۰۰ نمونه‌ی bootstrap (با seed ثابت) از اپیزودهای ارزیابی‌شده. این عدد در متریک‌ها، رکورد فشرده‌ی هر نسل و خط لاگ PPO (`fitness=0.123±0.190`) دیده می‌شود. تجمیع متریک‌های هر اپیزود به `metrics.episode_fitness_fields` منتقل شد تا bootstrap و عدد اصلی از یک تعریف استفاده کنند. **ارزیابی stochastic حفظ شد:** commit قبلی `7ab229a` با مطالعه‌ی ۳ seed × ۶ کاندید نشان داده بود که برچسب‌های argmax قطعی با عوض شدن seed آموزش جابه‌جا می‌شوند (همبستگی رتبه‌ای SR در حالت stochastic حدود ۰٫۶۶). اندازه‌گیری من هم نشان داد که بخش اصلی نویز از تنوع سناریوهاست، نه از نمونه‌گیری عمل. |
| M1 | اصلاح شد | `4bd610b` | پیش‌فرض `highway_warm_start` در `RaiseRunConfig`، `ClosedLoopConfig`، `Stage2Config` و `Stage3Config` حالا `False` است. با `--highway-warm-start` می‌شود دوباره فعالش کرد. تست plumbing در `test_pipeline` حالا مقدار `True` می‌فرستد تا اگر flag جایی گم شود، تست بشکند. |
| L1 | اصلاح شد | `db36e50` | مقدار fallback در `adapter` برای `highway_eval_deterministic` حالا `False` است، هم‌سو با هر دو config. |
| L2 | اصلاح شد | `5059a77` | `SOFT_PROGRESS_M = 400.0` به `objective_constants` منتقل شد و `_eval_metrics` از آن استفاده می‌کند. |
| L3 | بدون تغییر | — | همان‌طور که در بخش ۲ آمد، watchdog بعد از سقف زمانی هر فراخوانی (بخش ۲، M1) فقط برای فریزهای native لازم است. محدودیتش در docstring مستند است. |
| L4 | اصلاح شد | `766a0e4` | وقتی `device=cuda` خواسته شود ولی آموزش روی CPU انجام شود، یک بار در لاگ INFO نوشته می‌شود و نام `RAISE_HIGHWAY_FORCE_CUDA` را می‌آورد. |

**فرمول جدید fitness:**

</div>

```text
v_eff = speed_p10 ;  v_eff < 3.75  ->  F = -10
gate     = sigmoid(6 · (v_eff − 17.5) / 15)
survival = SR − 2·CR − TR
quality  = 0.35·tanh(PL/800) + 0.25·tanh(mean_speed/25)
F = gate·survival + max(0, survival)·quality − lag − degeneracy − low_speed
    (degeneracy now includes lane-lock below V_TARGET at any SR)
```

<div dir="rtl">

**شواهد بعد از اصلاح ۱ — always-SLOWER دوباره اندازه‌گیری شد** (همان ۶ seed ارزیابی، holdout با density=1.5، حالا ۳۰ اپیزود؛ ستون «قدیم» همان داده‌ها با فرمول قبلی است):

</div>

```text
seed  SR    CR    progress  speed  laneΔ   F (old)  F (new)  fitness_se
425   0.50  0.50  538 m     20.6   0.000    0.375   -0.524    0.192
426   0.73  0.27  667 m     20.4   0.000    0.750    0.081    0.249
427   0.77  0.23  668 m     20.5   0.000    0.811    0.196    0.249
428   0.63  0.37  597 m     20.5   0.000    0.584   -0.225    0.219
429   0.50  0.50  524 m     20.7   0.000    0.371   -0.524    0.215
430   0.43  0.57  494 m     20.7   0.000    0.258   -0.673    0.171

old formula: mean = +0.525  sd = 0.206
new formula: mean = -0.278  sd = 0.325   range [-0.673, +0.196]
SR sd across seeds: 0.175 (n=20)  ->  0.125 (n=30)
reference "good" policy (SR 0.9, 24 m/s, weaving):  old 1.212  new 0.924
```

<div dir="rtl">

- **معیار اول برآورده شد:** میانگین always-SLOWER از ‎+۰٫۴۹ (n=۲۰) به ‎−۰٫۲۸ رسید. فقط seedهایی که در آن‌ها SLOWER به‌طور اتفاقی ۷۳–۷۷٪ زنده می‌ماند کمی مثبت‌اند (۰٫۰۸ و ۰٫۲۰).
- **معیار دوم برآورده نشد:** انحراف معیار fitness بین seedها از ۰٫۲۱ به ۰٫۳۳ **بالا** رفت، با این‌که پراکندگی خود SR با n=۳۰ کم شد (۰٫۱۷۵ → ۰٫۱۲۵، تقریباً همان ضریب √(20/30) مورد انتظار). دلیل: فرمول جدید عمداً به SR حساس‌تر است (شیب fitness نسبت به SR تقریباً ۲٫۳ به جای ۱٫۶). فاصله‌ی سیگنال هم بزرگ‌تر شده است: اختلاف policy خوب و SLOWER از ۰٫۶۹ به ۱٫۲۰ رسید. پس نسبت سیگنال به نویز کمی بهتر شده (حدود ۳٫۳ به ۳٫۷ انحراف معیار)، ولی نه به‌طور چشمگیر.
- **نتیجه‌ی عملی:** `fitness_se` هر ارزیابی بین ۰٫۱۷ و ۰٫۲۵ است. اختلاف کمتر از حدود ۰٫۴ بین دو کاندید (یا بین نسل‌ها) در حد نویز ارزیابی است و نباید در پایان‌نامه به‌عنوان بهبود گزارش شود. برای ادعای Stage III، ارزیابی ۳۰ اپیزودی کافی نیست. این موضوع در بخش ۸ (بودجه‌ی Stage III) پیگیری می‌شود.

**شواهد بعد از اصلاح ۲ — منحنی یادگیری (M2)** پاداش seed، آموزش از صفر، بدون warm start، holdout با ۲۰ اپیزود، eval stochastic، seed=425:

</div>

```text
steps   SR    CR    progress  speed  p10    laneΔ   F (old)  F (new)  wall
 6000   0.00  1.00  190 m     24.1   21.5   0.047   -0.562   -1.661   286 s
12000   0.00  1.00  122 m     25.8   25.0   0.074   -0.706   -1.905   478 s
24000   0.00  1.00  108 m     25.8   24.8   0.114   -0.707   -1.896   787 s
```

<div dir="rtl">

- با پاداش seed، دو برابر کردن بودجه هیچ کمکی نمی‌کند: policy سریع‌تر می‌راند، بیشتر لاین عوض می‌کند و **زودتر** تصادف می‌کند (پیشرفت ۱۹۰ → ۱۲۲ → ۱۰۸ متر). مشکل از خود پاداش است (جریمه‌ی ثابت ۲۰- برای تصادف در برابر پاداش متراکم سرعت/پیشرفت)، نه از کمبود بودجه. پس افزایش K2 فعلاً توجیه ندارد. وظیفه‌ی حلقه‌ی LLM دقیقاً پیدا کردن پاداشی است که این مشکل را نداشته باشد.
- این با بخش ۱ سازگار است: آن‌جا هم با seed آموزشی ۴۲۵ پاداش seed تقریباً همیشه تصادف می‌کرد (SR بین ۰ و ۰٫۲۰) و فقط با seed ۴۲۶ به always-SLOWER رسید. پس نتیجه‌ی پاداش seed به seed آموزش وابسته است، ولی با هیچ seedی و هیچ بودجه‌ای تا ۲۴k به policy خوبی نمی‌رسد.
- فرمول جدید این policy را خیلی شدیدتر جریمه می‌کند (‎−۱٫۹ به جای ‎−۰٫۷). پس در رتبه‌بندی، «سریع ولی همیشه تصادف» دیگر به «کند و نیمه‌امن» نزدیک نیست.

**تست‌ها:** مجموعه‌ی سریع (`pytest -m "not slow"`) با ۲۴۲ تست و تست کند `test_wrapper_step` هر دو پاس شدند. تست‌های جدید: `test_crash_prone_slow_policy_scores_non_positive`، `test_crash_costs_more_than_survival_earns`، `test_lane_locked_below_target_penalized_at_moderate_sr`، `test_soft_success_is_diagnostic_only`، `test_fitness_bootstrap.py` (۴ تست)، `test_warm_start_off_by_default`، `test_cuda_fallback_logged_once`.

**هزینه‌ی زمانی:** ارزیابی holdout حدود ۳ ثانیه برای هر اپیزود است. پس ۱۰ اپیزود اضافه در Stage II حدود ۳۰ ثانیه به هر کاندید اضافه می‌کند؛ با پیش‌فرض N=6 و G=4 در کل حدود ۱۲ دقیقه.

</div>
