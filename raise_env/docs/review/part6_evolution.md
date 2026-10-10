<div dir="rtl">

# بررسی بخش ۶: موتور تکامل

**دامنه:**

- `raise_core/explore.py` (~۱۰۸۰ خط؛ بخش‌های تکامل: `_next_generation`، `_fill_valid`، `_try_crossover`، `_try_mutate`، `_fill_random_batch`، `_build_reflection`)
- `raise_core/selection.py` (~۷۰ خط)
- `raise_core/raise_loop/evolve_rank.py` (~۹۵ خط)
- `raise_core/raise_loop/proxy_feedback.py` (~۸۰ خط)

نقاط اتصال هم بررسی شد: `raise_loop/runner.py` (حلقه‌ی نسل‌ها، `_needs_training`، seed ارزیابی)، `pipeline._closed_loop_config` (تعداد عملگرها)، `parallel_label.py`، و `scripts/plot_raise_run.py` (نمودار best fitness).

**روش:**

- کد خط به خط خوانده شد.
- تعداد واقعی عملگرها برای N=۴، ۶، ۸ و ۱۰ از خود `RaisePipeline._closed_loop_config` گرفته شد.
- `rank_population_for_evolution` با یک کاندید ردشده در Score1 اجرا شد.
- اثر elitism با شبیه‌سازی Monte Carlo (۲۰۰ هزار تکرار) روی پیش‌فرض‌های واقعی اندازه‌گیری شد (N=۶، G=۴، انحراف معیار نویز ارزیابی ۰٫۲، یعنی همان `fitness_se` اندازه‌گیری‌شده در بخش ۵).
- در workspace هیچ خروجی اجرای واقعی (`pop_epoch_*.json`) وجود ندارد. پس هر عددی که به رفتار LLM واقعی وابسته است (مثلاً نرخ fallback یا کد تکراری) اندازه‌گیری نشده و در متن با همین قید آمده است.
  - **اصلاحیه (بخش ۳):** اجراهای قدیمی Groq در `results/` هستند. در شش جمعیت نهایی آن‌ها (۳۶ کاندید) هیچ ژنوم تکراری و هیچ fallback نبود؛ پس M1 در عمل به‌ندرت فعال می‌شود. جزئیات در `part3_llm_prompts.md`.

**سطح‌بندی یافته‌ها:** همان بخش‌های قبل.

- **H (بالا):** روی اعتبار نتایج پایان‌نامه یا رتبه‌بندی کاندیدها اثر مستقیم دارد.
- **M (متوسط):** باگ یا سوگیری در مسیر واقعی، یا هزینه‌ی محاسباتی قابل‌توجه.
- **L (پایین):** تمیزکاری، ناسازگاری مستندات، یا کد مرده.

</div>

---

<div dir="rtl">

## ۱. خلاصه‌ی معماری

جریان یک نسل با پیش‌فرض‌های واقعی (`RaiseRunConfig`: N=۶، G=۴، `elitism=True`):

</div>

```text
ClosedLoopRunner, generation g
  score_population(pop)                    # Score1, sorted by Score1
  to_label = [c for c in pop if fitness is None]     # elite is NOT retrained
  eval_seed_base = seed + g*1000           # shared by the candidates trained now
  PPO train + holdout eval of to_label     # ~478 s train + ~90 s eval each
  ranked = rank_population_for_evolution(pop, mode="scalar")
             # trained by fitness (tie: Score1), untrained after, by Score1
  archive.add_population(ranked); reflection = population table + trend
  if g+1 < G:
      next = [ranked[0]]                                  # elite (stale fitness)
           + 2 x crossover(ranked[0], ranked[1])          # same two parents
           + 3 x mutation(ranked[3], ranked[4], ranked[5])  # lower half
           + 0 x random                                   # N=6 -> no random slot
      each slot: up to 17 LLM attempts, then fallback clone
```

<div dir="rtl">

**fallbackها** (وقتی ۱۷ تلاش LLM برای یک اسلات شکست بخورد یا درخواست از بودجه‌ی توکن بزرگ‌تر باشد):

| عملگر | جایگزین |
|---|---|
| crossover | کپی کد `ranked[0]` |
| mutation | کپی کد همان والد |
| random و Gen0 | تابع seed |

</div>

---

<div dir="rtl">

## ۲. یافته‌ها

### H1. elite با fitness کهنه و «بیشینه‌ی نمونه‌های نویزی» به نسل بعد می‌رود؛ منحنی best fitness حتی بدون هیچ پیشرفتی بالا می‌رود

**محل:** `explore._next_generation` (خط ۹۴۹: `next_pop.append(top_a)`)؛ `runner._needs_training` (کاندید دارای `fitness` دوباره آموزش و ارزیابی نمی‌شود)؛ `runner` خط ۴۲۶ (`highway_eval_seed_base = seed + g*1000`)؛ `epoch_rec["evolve_best_fitness"]` و نمودار «Best fitness» در `plot_raise_run.plot_closed_loop_epochs`.

**مشکل:**

1. **fitness انتخاب‌شده از نویز:** elite همان کاندیدی است که بیشترین fitness را در نسل خودش گرفته، یعنی بیشینه‌ی چند نمونه‌ی نویزی (با `fitness_se` بین ۰٫۱۷ و ۰٫۲۵). عدد ذخیره‌شده‌اش به‌طور سیستماتیک بالاتر از کیفیت واقعی‌اش است (winner's curse).
2. **سناریوهای متفاوت:** elite روی سناریوهای ارزیابی نسل قبل نمره گرفته، ولی فرزندان روی سناریوهای نسل فعلی. بخش ۵ دقیقاً به این دلیل seed ارزیابی را داخل هر نسل مشترک کرد که مقایسه منصفانه باشد؛ elite از این قاعده بیرون است.
3. **منحنی یکنوا:** fitness elite هرگز کم نمی‌شود. پس `evolve_best_fitness` ساختاراً غیرنزولی است و با بیشینه‌گرفتن روی نمونه‌های بیشتر بالا می‌رود، حتی وقتی هیچ کاندیدی بهتر نشده باشد.

**شواهد: Monte Carlo** با N=۶، G=۴، elite به اضافه‌ی ۵ فرزند در هر نسل، و انحراف معیار کل نویز ۰٫۲. نویز به دو بخش تقسیم شد: «مشترک» برای کاندیدهای یک نسل (اثر seed مشترک) و «اختصاصی» برای هر کاندید.

</div>

```text
children's true gain = 0 (no real progress at all)
common SD  policy             reported best gain   true gain   gen0 elite never replaced   child takes the top / gen
0.00       stale elite (now)        +0.124           0.000              29%                        33%
0.10       stale elite (now)        +0.145           0.000              28%                        34%
0.14       stale elite (now)        +0.164           0.000              27%                        35%
0.00-0.14  re-evaluate elite        +0.001           0.000               0%                        83%

children truly +0.1 better than the elite every generation
0.00       stale elite (now)        +0.275          +0.171              12%                        57%
0.14       stale elite (now)        +0.306          +0.158              14%                        53%
0.00-0.14  re-evaluate elite     +0.271..0.277   +0.275..0.282          0%                      92-94%
```

<div dir="rtl">

- **بدون هیچ پیشرفت واقعی**، طرح فعلی در ۴ نسل ‎+۰٫۱۲ تا ‎+۰٫۱۶ «بهبود» در best fitness گزارش می‌کند. این عدد هم‌اندازه‌ی یک `fitness_se` است و روی نمودار پایان‌نامه شبیه یادگیری دیده می‌شود.
- **با پیشرفت واقعی**، elite کهنه جلوی فرزندان بهتر را می‌گیرد. فقط ۰٫۱۶ تا ۰٫۱۷ از ۰٫۲۸ پیشرفت واقعی به elite نهایی می‌رسد، و در هر نسل فقط ۵۳ تا ۵۷ درصد مواقع یک فرزند بهتر جای elite را می‌گیرد (در برابر ۹۲ تا ۹۴ درصد وقتی elite دوباره ارزیابی شود).
- عدد گزارش‌شده در حالت فعلی حدود ۰٫۱۲ بیشتر از پیشرفت واقعی است.

**نکته‌ی کاهنده:** Stage III همه‌ی جمعیت نهایی، از جمله elite، را از نو آموزش و ارزیابی می‌کند. پس انتخاب برنده‌ی نهایی مستقیماً از این خطا آسیب نمی‌بیند (winner's curse خود Stage III در بخش ۸ بررسی می‌شود). آسیب اصلی در دو جاست:

- منحنی همگرایی Stage II و ادعای «حلقه بهبود می‌دهد»؛
- فشار انتخاب داخل حلقه، که فرزندان خوب را دور می‌اندازد.

**پیشنهاد اصلاح:**

- **(الف) پیشنهادی:** در هر نسل، `model.zip` ذخیره‌شده‌ی elite (کلید `checkpoint_path` در متادیتا) را **فقط ارزیابی** کن، بدون آموزش دوباره، روی همان holdout و `eval_seed_base` نسل فعلی. هزینه حدود ۹۰ ثانیه در هر نسل است، در برابر حدود ۸ دقیقه برای آموزش یک کاندید. fitness قبلی در متادیتا به‌عنوان تاریخچه بماند (`fitness_history`).
- **(ب):** elitism را خاموش کن و بهترین‌های همه‌ی نسل‌ها را فقط در archive نگه دار. این ساده‌تر است ولی ممکن است بهترین ژنوم از جمعیت حذف شود.
- **(ج) حداقل:** `fitness_generation` را روی هر کاندید ثبت کن و منحنی best fitness را فقط از ارزیابی‌های تازه‌ی همان نسل رسم کن.

**معیار موفقیت:** با trainer جعلی‌ای که fitness را از «کیفیت ثابت به اضافه‌ی نویز» می‌سازد، میانگین بهبود گزارش‌شده بدون پیشرفت واقعی تقریباً صفر شود. همچنین در هر نسل، fitness همه‌ی کاندیدهای رتبه‌بندی‌شده از همان `eval_seed_base` نسل آمده باشد.

</div>

---

<div dir="rtl">

### H2. الگوریتمی که اجرای پیش‌فرض انجام می‌دهد با الگوریتم مستندشده فرق دارد: در N=۶ هیچ random restartی وجود ندارد

**محل:** `pipeline._closed_loop_config` (خطوط ۱۶۵ تا ۱۸۰)؛ `explore._next_generation` (خطوط ۹۳۰ تا ۹۴۲)؛ `spec.md` خطوط ۲۴۶ تا ۲۴۸؛ `README_RAISE.md` خطوط ۶۲ تا ۶۵؛ توضیح `StageIConfig.keep_runtime_elite`.

**شواهد:** تعداد واقعی عملگرها از خود `RaisePipeline._closed_loop_config`:

</div>

```text
N=4:  elite=1 crossover=2 mutation=1 random=0   mutation parents = ranks 3-4
N=6:  elite=1 crossover=2 mutation=3 random=0   mutation parents = ranks 4-6, rank 3 never used   <- default
N=8:  elite=1 crossover=2 mutation=4 random=1   ranks 3-4 never used
N=10: elite=1 crossover=2 mutation=4 random=3   ranks 3-5 never used
```

<div dir="rtl">

**مشکل:**

- مستندات می‌گویند هر نسل «crossover، mutation و random restart» دارد، و توضیح کد از نسبت «paper-faithful ۲/۴/۲» حرف می‌زند. ولی pipeline تعداد crossover و mutation را ثابت روی `min(2, N)` و `min(4, N−2)` می‌گذارد و فقط باقی‌مانده را به random می‌دهد. elitism هم یک اسلات از random یا mutation برمی‌دارد.
- با پیش‌فرض N=۶ **هیچ ژنوم تازه‌ای بعد از Gen0 وارد جمعیت نمی‌شود**. همه‌ی جمعیت از نسل‌های قبل مشتق می‌شوند. سؤال «آیا تنوع حفظ می‌شود» از برنامه‌ی بررسی، در پیکربندی پیش‌فرض یعنی «هیچ سازوکاری برایش نیست».
- رتبه‌ی ۳ (و در N بزرگ‌تر رتبه‌های میانی) نه والد crossover است و نه والد mutation. پس دومین و سومین کاندیدهای خوب عملاً دور ریخته می‌شوند.
- در CLI و `RaiseRunConfig` هیچ گزینه‌ای برای تعداد عملگرها نیست. `ClosedLoopConfig` این فیلدها را دارد، ولی pipeline آن‌ها را بازنویسی می‌کند.

**اثر:** اگر پایان‌نامه الگوریتم را «۲/۴/۲ با elitism» توصیف کند، با اجرای واقعی نمی‌خواند.

**پیشنهاد اصلاح:**

- **(الف) پیشنهادی:** سهم عملگرها را صریح و قابل تنظیم کن (`--n-crossover`، `--n-mutation`، `--n-random`) با پیش‌فرضی که برای N=۶ دست‌کم یک random دارد، مثلاً ۱ elite، ۲ crossover، ۲ mutation و ۱ random. سهم واقعی هر نسل هم در `epochs.jsonl` ثبت شود.
- **(ب):** پیش‌فرض‌ها همین بمانند، ولی `spec.md` و README دقیقاً همین را بنویسند («در N=۶: ۲ crossover، ۳ mutation، ۱ elite، بدون random»).

</div>

---

<div dir="rtl">

### M1. ژنوم‌های تکراری و کپی‌های fallback با هزینه‌ی کامل دوباره آموزش می‌بینند

**محل:** `explore._clone_valid` و fallbackهای `_next_generation` (خطوط ۹۶۶ تا ۹۹۳)؛ `_seed_fallback_candidate`؛ `runner._needs_training`؛ `selection.is_same_genome` (هیچ‌جا استفاده نمی‌شود).

**مشکل:** در کل حلقه هیچ تشخیص تکراری وجود ندارد. منابع تکرار:

- دو اسلات crossover **دقیقاً یک prompt** می‌فرستند (همان دو والد و همان کارت)، با `temperature=0.4`.
- fallback برای crossover کد `ranked[0]` را کپی می‌کند. وقتی elitism روشن است، این یعنی نسخه‌ی دوم همان elite.
- fallback برای mutation کد والدی را کپی می‌کند که در نیمه‌ی پایین بوده، یعنی ژنومی که می‌دانیم ضعیف است.
- fallback برای random و Gen0 تابع seed است. بخش ۵ نشان داد پاداش seed با هر بودجه‌ای تا ۲۴k همیشه تصادف می‌کند (SR=۰).

هر کپی یک کاندید تازه با `fitness=None` است، پس دوباره کامل آموزش داده می‌شود: حدود ۴۷۸ ثانیه آموزش و ۹۰ ثانیه ارزیابی. اگر API برای یک نسل از دسترس خارج شود، تا ۵ کپی ساخته می‌شود و حدود ۴۵ دقیقه‌ی PPO هدر می‌رود. علاوه بر این، آموزش دوباره‌ی یک ژنوم یک «نمونه‌ی نویزی دیگر» از همان ژنوم است و احتمال اول‌شدنش را بی‌دلیل بالا می‌برد، با همان سازوکار H1.

**نرخ واقعی اندازه‌گیری نشده است**، چون اجرای واقعی‌ای در workspace نیست. ولی مسیرها در کد قطعی‌اند.

**پیشنهاد اصلاح:**

- یک «اثر انگشت» از کد بساز که از AST نرمال‌شده گرفته شود، تا کامنت و فاصله‌گذاری اثری نداشته باشند.
- فرزند LLM که با ژنومی در جمعیت یا archive یکسان است، تلاش ناموفق حساب شود و دوباره تولید شود.
- کپی fallback از ژنومی که قبلاً ارزیابی شده، دوباره آموزش نبیند. ارزیابی همان ژنوم را بگیرد (با اصلاح H1-الف، فقط دوباره ارزیابی شود) و `duplicate_of` ثبت شود.
- تعداد ژنوم‌های یکتای هر نسل در `epochs.jsonl` ثبت شود (همراه با L7).

</div>

---

<div dir="rtl">

### M2. رتبه‌بندی `hybrid` با یک کاندید ردشده در Score1 کاملاً از کار می‌افتد؛ حالت‌های `score1` و `hybrid` روی سیگنالی کار می‌کنند که اعتبارش نشان داده نشده

**محل:** `evolve_rank.rank_population_for_evolution` (خطوط ۷۷ تا ۹۵).

**مشکل:** کاندیدی که در Score1 hard-reject شود `score=-inf` می‌گیرد، ولی چون `reward_fn` دارد، باز هم آموزش می‌بیند و برچسب می‌گیرد. در `hybrid`، کمینه‌ی Score1 برابر `-inf` می‌شود و `(v − lo) / (hi − lo)` برابر `inf/inf = NaN` است. پس کلید **همه‌ی** کاندیدها NaN می‌شود و `sorted` همان ترتیب ورودی را برمی‌گرداند.

</div>

```text
input  : ['bad', 's1_rejected', 'mid', 'good']     fitness: bad=-1.5 rej=0.5 mid=0.1 good=0.9
hybrid : ['bad', 's1_rejected', 'mid', 'good']     <- unchanged input order; 'bad' becomes crossover parent
without the rejected candidate: ['good', 'mid', 'bad']
```

<div dir="rtl">

در اجرای واقعی، ورودی به ترتیب Score1 است. پس `hybrid` بی‌صدا به `score1` تبدیل می‌شود. علاوه بر این، بخش ۴ نشان داد Score1 هیچ رابطه‌ی اثبات‌شده‌ای با fitness ندارد و فقط فیلتر سلامت است. پس `score1` و `hybrid` اصلاً نباید معیار انتخاب والد باشند.

**پیشنهاد اصلاح:**

- **(الف) پیشنهادی، هم‌سو با ساده‌سازی:** حالت‌های `score1` و `hybrid` و گزینه‌های CLI آن‌ها حذف شوند. فقط ترتیب fitness بماند، و Score1 فقط به‌عنوان tie-break و برای ترتیب کاندیدهای بدون برچسب.
- **(ب):** حالت‌ها بمانند، ولی Score1 نامتناهی پیش از نرمال‌سازی کنار گذاشته شود (کمینه‌ی متناهی، یا ۰ برای `-inf`)، و یک تست اضافه شود.

</div>

---

<div dir="rtl">

### M3. mutation فقط روی نیمه‌ی پایین است؛ بهترین ژنوم هیچ‌وقت با شواهد خودش اصلاح نمی‌شود

**محل:** `explore._next_generation` (خط ۹۴۴: `lower = ranked[len(ranked) // 2:]`).

**مشکل:**

- تنها راه بهره‌برداری از بهترین ژنوم، crossover آن با رتبه‌ی دوم است. mutation همیشه روی ضعیف‌ترین‌ها انجام می‌شود، که شامل کاندیدهای DQ‌شده (fitness برابر ۱۰−) و کاندیدهایی که آموزششان شکست خورده هم هست.
- حالا که کارت بازخورد برای mutation شواهد غنی می‌فرستد (تفاوت با والد، رفتار، مؤلفه‌های پاداش، تاریخچه‌ی نسب)، مفیدترین فراخوانی احتمالاً «mutation بهترین ژنوم با تشخیص ضعف‌های خودش» است.
- این طراحی از مقاله آمده است («mutate lower performers using their weakness»)، پس باگ نیست. ولی سؤال برنامه‌ی بررسی («آیا جهش روی نیمه‌ی پایین بهترین انتخاب است؟») جوابش احتمالاً منفی است.

**پیشنهاد:** این تصمیم با توست، چون منطق انتخاب تکاملی را عوض می‌کند. یک گزینه‌ی `mutation_parents` اضافه شود:

- `lower`: رفتار فعلی؛
- `rank_weighted`: احتمال انتخاب متناسب با رتبه؛
- `tournament`: tournament دوتایی.

پیش‌فرض فعلاً بدون تغییر بماند تا یک مقایسه‌ی کنترل‌شده انجام شود.

</div>

---

<div dir="rtl">

### L1. بودجه‌ی تلاش LLM برای هر اسلات جدا حساب می‌شود و بزرگ است

**محل:** `StageIConfig.max_invalid_replacements=16`؛ `_fill_valid(needed=1, max_extra=16)` برای هر اسلات crossover و mutation.

در Gen0 بودجه‌ی اضافه بین همه‌ی اسلات‌ها مشترک است (`remaining + 16` تلاش). ولی در نسل‌های بعد **هر اسلات** ۱۷ تلاش دارد. بدترین حالت با پیش‌فرض‌ها ۵ × ۱۷ = ۸۵ فراخوانی LLM در هر نسل است. این زمان و سهمیه‌ی Groq می‌سوزاند، در حالی که تولید پشت‌سرهم کد نامعتبر برای یک prompt معمولاً با تلاش بیشتر درست نمی‌شود.

**پیشنهاد:** یک بودجه‌ی مشترک برای هر نسل (مثلاً N + ۸)، یا سقف کوچک‌تر برای هر اسلات (مثلاً ۴).

</div>

---

<div dir="rtl">

### L2. `global_best` با Score1 انتخاب می‌شود، checkpoint می‌شود، و هیچ‌جا استفاده نمی‌شود

**محل:** `runner` خطوط ۳۷۰ تا ۳۷۷ و ۲۹۲ و ۳۰۹؛ `explore.StageIEvolver.global_best`.

بهترینِ Score1 است، نه fitness، و هیچ مصرف‌کننده‌ای ندارد. فقط در checkpoint نوشته و خوانده می‌شود. **پیشنهاد:** حذف شود (یا اگر لازم است، بر اساس fitness باشد).

</div>

---

<div dir="rtl">

### L3. `_best_fitness_trend` در checkpoint نیست

**محل:** `explore._build_evidence_reflection`.

بعد از ادامه‌ی اجرا از checkpoint، روند از صفر شروع می‌شود، در حالی که متن reflection بازیابی می‌شود. پس دو عدد ناسازگار به LLM می‌رسد. همین روند از elite کهنه هم ساخته می‌شود (H1). **پیشنهاد:** روند از `history` (همان `evolve_best_fitness` هر نسل) ساخته شود، نه از state جداگانه.

</div>

---

<div dir="rtl">

### L4. پیش‌فرض‌های ناسازگار برای elitism و تعداد عملگرها

**محل:**

- `StageIConfig.keep_runtime_elite=False` با توضیح «paper-faithful»، در حالی که `ClosedLoopConfig` و `RaiseRunConfig` مقدار `True` دارند.
- `runner` اگر جمع سهم‌ها با N برابر نباشد، فقط هشدار می‌دهد و ۲/۴/باقی‌مانده را جایگزین می‌کند.

**پیشنهاد:** یک منبع واحد برای پیش‌فرض‌ها (همراه با H2)، و خطا به جای جایگزینی بی‌صدا.

</div>

---

<div dir="rtl">

### L5. import حلقوی `explore` ↔ `domains.highway`

**محل:** `explore.py` خطوط ۲۶ و ۲۷ ← `domains/highway/__init__` ← `pack` ← `adapter` ← `refine` ← `from raise_core.explore import RewardCandidate`.

اگر `raise_core.explore`، `raise_core.selection` یا `raise_core.refine` **اولین** ماژولی باشد که import می‌شود، `ImportError` می‌دهد (هر سه آزموده شد). ماژول‌های `raise_loop/*` سالم‌اند، چون `raise_loop/__init__` اول `domains` را بار می‌کند. این مشکل قبل از کارت بازخورد هم وجود داشت (commit `d8d5ad9` با همان خطا آزموده شد). فقط به این دلیل دیده نمی‌شود که نقاط ورود همیشه اول `domains` را import می‌کنند.

**پیشنهاد:** `RewardCandidate` به ماژول سبک `raise_core/candidate.py` منتقل شود. `refine`، `selection`، `evolve_rank` و `proxy_feedback` از آن‌جا import کنند، و `explore` برای سازگاری آن را دوباره export کند.

</div>

---

<div dir="rtl">

### L6. random restart کارت «reference» بهترین ژنوم را می‌گیرد

**محل:** `explore._make_card` (وقتی subject ندارد، بهترین fitness آرشیو را می‌آورد).

این خواسته‌ی صریح کار کارت بازخورد بود («wire into random calls»). ولی نقش random restart اکتشاف است، و نشان‌دادن بهترین ژنوم فعلی، ژنوم‌های تازه را به سمت همان ژنوم می‌کشد. **پیشنهاد:** گزینه‌ی `feedback.random_calls` (پیش‌فرض: روشن، همان رفتار فعلی) تا بشود اثرش را جدا سنجید.

</div>

---

<div dir="rtl">

### L7. هیچ معیار تنوعی ثبت نمی‌شود

برای جواب‌دادن به سؤال «آیا تنوع جمعیت حفظ می‌شود» داده‌ای نداریم. **پیشنهاد:** در `epochs.jsonl` این‌ها ثبت شوند:

- `n_unique_genomes` (از اثر انگشت M1)؛
- میانگین فاصله‌ی Jaccard مجموعه‌ی مؤلفه‌های پاداش بین جفت‌ها (از `feedback.code_structure`، بدون هزینه‌ی اضافه).

</div>

---

<div dir="rtl">

### L8. نام‌ها و aliasهای زائد در `selection.py`

`candidate_fitness` همان `candidate_nav_scalar` است. نام‌های `navigation_scalar*` از دامنه‌ی ناوبری قدیمی مانده‌اند، و `selection_scalar` هم alias همان `fitness` است. **پیشنهاد:** یک تابع `candidate_fitness` بماند و بقیه حذف شوند.

</div>

---

<div dir="rtl">

## ۳. مواردی که بررسی و تأیید شد

- **جمع سهم عملگرها:** `StageIEvolver` اگر جمع با N برابر نباشد، خطا می‌دهد. `_next_generation` اندازه‌ی جمعیت را با `assert` چک می‌کند.
- **ترتیب scalar:** کاندیدهای آموزش‌دیده با fitness و tie-break با Score1. کاندیدهای بدون برچسب (آموزش شکست‌خورده) بعد از همه می‌آیند. والد crossover همیشه از کاندیدهای آموزش‌دیده است، مگر وقتی کمتر از دو کاندید برچسب داشته باشند.
- **mutation والد بدون ارزیابی:** متن قدیمی صریحاً می‌گوید «ارزیابی holdout ندارد». با کارت، بخش‌های بدون داده با دلیل «unavailable» علامت می‌خورند.
- **حلقه هیچ‌وقت به‌خاطر LLM نمی‌میرد:** خطاهای قابل بازیابی شمرده می‌شوند. `PromptBudgetError` فوراً به fallback می‌رود. خطای غیرقابل بازیابی بالا می‌رود.
- **شناسه‌ها:** `restore_id_counter` بعد از ادامه‌ی اجرا از همه‌ی شناسه‌های دیده‌شده جلو می‌زند (جزئیات در بخش ۷).
- **seed مشترک:** همه‌ی کاندیدهایی که در یک نسل آموزش می‌بینند روی همان سناریوها ارزیابی می‌شوند (به‌جز elite، یعنی H1).
- **کارت بازخورد:** در کار قبلی با ۳۳ تست پوشش داده شده است (بودجه‌ی توکن، تاریخچه‌ی نسب، لاگ هر فراخوانی).

</div>

---

<div dir="rtl">

## ۴. موارد خارج از دامنه

| مورد | بخش مقصد |
|---|---|
| winner's curse در انتخاب برنده‌ی Stage III (بیشینه‌ی N ارزیابی ۳۰ اپیزودی) | ۸ |
| آموزش کاندیدهای ردشده در Score1 (`_needs_training`) | ۷ |
| درستی ادامه‌ی اجرا در همه‌ی فازها | ۷ |
| backoff و retry لایه‌ی کلاینت LLM در قطعی طولانی | ۳ (پیگیری) |

</div>

---

<div dir="rtl">

## ۵. جدول اولویت

| شناسه | سطح | خلاصه | هزینه‌ی اصلاح |
|---|---|---|---|
| H1 | بالا | elite کهنه: ‎+۰٫۱۲ تا ‎+۰٫۱۶ «بهبود» بدون پیشرفت واقعی؛ فرزندان بهتر کنار گذاشته می‌شوند | متوسط (ارزیابی دوباره‌ی elite) |
| H2 | بالا | N=۶: بدون random، رتبه‌ی ۳ بی‌استفاده؛ خلاف مستندات | کم |
| M1 | متوسط | ژنوم تکراری و کپی fallback دوباره آموزش می‌بینند | کم تا متوسط |
| M2 | متوسط | `hybrid` با Score1 برابر ‎−inf به NaN می‌رسد؛ حالت‌های Score1 بی‌پشتوانه | کم |
| M3 | متوسط | mutation فقط نیمه‌ی پایین (تصمیم طراحی، با تو) | کم (گزینه) |
| L1–L8 | پایین | بودجه‌ی تلاش، `global_best` مرده، روند، پیش‌فرض‌ها، import حلقوی، کارت random، معیار تنوع، aliasها | کم |

**ترتیب پیشنهادی اصلاح:** L5 (تا تست‌ها مستقل از ترتیب import شوند) ← H1 ← M1 ← H2 ← M2 ← L1 تا L4، L7 و L8. M3 و L6 فقط به‌صورت گزینه و با پیش‌فرض فعلی.

</div>

---

<div dir="rtl">

## ۶. وضعیت اصلاحات

همه‌ی یافته‌ها به ترتیب پیشنهادی اصلاح شدند، روی شاخه‌ی `simplify/highway-core`. برای H2 گزینه‌ی «قابل تنظیم» و برای M2 گزینه‌ی «حذف» انتخاب شد. M3 و L6 فقط به‌صورت گزینه اضافه شدند و پیش‌فرض‌ها همان رفتار قبلی است.

| شناسه | وضعیت | commit | چه تغییر کرد |
|---|---|---|---|
| L5 | اصلاح شد | `dc808cd` | `RewardCandidate` به `raise_core/candidate.py` منتقل شد و `explore` آن را دوباره export می‌کند. `domains/highway/__init__` حالا خالی است و registry مستقیم به `domains.highway.pack` اشاره می‌کند. تست جدید `test_import_order.py` هر ماژول را در یک subprocess تازه به‌عنوان اولین import بار می‌کند. |
| H1 | اصلاح شد | `1f21ab1` | ژنومی که با elitism منتقل می‌شود دوباره آموزش نمی‌بیند. policy ذخیره‌شده‌اش روی seedهای holdout همان نسل دوباره ارزیابی می‌شود (`trainer.reevaluate`، بدون آموزش). ارزیابی‌های قبلی در `fitness_history` می‌مانند. اگر ارزیابی دوباره شکست بخورد، fitness قبلی با `fitness_stale` حفظ می‌شود و در `stale_fitness_ids` می‌آید. `epochs.jsonl` فیلد `reevaluated_ids` را هم دارد. |
| M1 | اصلاح شد | `b8076d9` | اثر انگشت ژنوم: SHA-1 درخت AST بدون docstring، پس فاصله‌گذاری و کامنت اثری ندارند. فرزند LLM که با ژنوم آرشیو یا هم‌نسل خودش یکی باشد، با دسته‌ی `duplicate` رد و دوباره تولید می‌شود. کپی fallback ارزیابی ژنوم مبدأ را به ارث می‌برد (`duplicate_of`) و فقط دوباره ارزیابی می‌شود، نه آموزش. ژنوم‌های تازه‌ی یکسان در یک نسل فقط یک بار آموزش می‌بینند. والد دوم crossover اولین ژنوم متفاوت با بهترین است. فیلدهای جدید: `n_unique_genomes` و `duplicate_ids`. |
| H2 | اصلاح شد (قابل تنظیم) | `921382c` | `operator_counts()` تنها منبع سهم عملگرهاست و شرط elite + crossover + mutation + random = N را نگه می‌دارد. پیش‌فرض: یک random (وقتی دست‌کم ۳ فرزند باشد) و بقیه نصف بین mutation (نیمه‌ی بزرگ‌تر) و crossover. پس در N=۶: ۱ elite، ۲ crossover، ۲ mutation و ۱ random. گزینه‌های `--n-crossover`، `--n-mutation` و `--n-random` اضافه شدند و اولین سهم تعیین‌نشده باقی‌مانده را می‌گیرد. جمع ناسازگار حالا خطا می‌دهد (CLI با کد ۲ خارج می‌شود). manifest فیلد `operators` را دارد و هر نسل `origin_counts` (شامل fallbackها) و `n_carried` را. |
| M2 | اصلاح شد (حذف) | `36d848f` | حالت‌های `score1` و `hybrid` و گزینه‌های `--evolve-rank` و `--evolve-rank-score1-weight` حذف شدند. ترتیب والدها فقط بر اساس fitness است. Score1 فقط tie-break و ترتیب کاندیدهای بدون شواهد Stage II است. |
| L1 | اصلاح شد | `741f452` | یک بودجه‌ی تلاش برای هر نسل که بین همه‌ی اسلات‌ها مشترک است (`RetryBudget`). بدترین حالت حالا «تعداد فرزندان + ۱۶» فراخوانی است، نه ۱۷ فراخوانی برای هر اسلات. |
| L2 | اصلاح شد | `741f452` | `global_best` از evolver، runner و checkpoint حذف شد. |
| L3 | اصلاح شد | `741f452` | روند بهترین fitness در reflection حالا از `evolve_best_fitness` همان `history` ساخته می‌شود. پس بعد از ادامه‌ی اجرا هم با متن reflection سازگار می‌ماند. |
| L4 | اصلاح شد | `921382c`، `741f452` | runner دیگر سهم‌ها را بی‌صدا جایگزین نمی‌کند (H2). پیش‌فرض `keep_runtime_elite` در `StageIConfig` هم مثل configهای اجرا `True` شد. |
| L6 | گزینه اضافه شد | `44f8034` | `FeedbackCardConfig.random_calls` و `--no-feedback-random-calls`. پیش‌فرض روشن است، یعنی همان رفتار قبلی. |
| L7 | اصلاح شد | `b8076d9`، `741f452` | `epochs.jsonl` حالا `n_unique_genomes` و `component_diversity` را دارد. دومی میانگین فاصله‌ی Jaccard مجموعه‌ی مؤلفه‌های پاداش بین جفت‌هاست (۰ یعنی همه یکسان). |
| L8 | اصلاح شد | `741f452` | در `selection.py` فقط `candidate_fitness` و `metrics_fitness` ماندند. `navigation_scalar*`، `candidate_nav_scalar` و `is_same_genome` حذف شدند. کلید داده‌ی `selection_scalar` عمداً ماند، چون نویسنده‌های متریک و فایل‌های نتایج قبلی از آن استفاده می‌کنند. |
| M3 | گزینه اضافه شد | `44f8034` | `--mutation-parents` با مقدارهای `lower`، `rank_weighted` یا `tournament` و پیش‌فرض `lower` (همان نیمه‌ی پایین). انتخاب تصادفی با seed اجرا و شناسه‌های رتبه‌بندی seed می‌شود، پس اجرای ادامه‌یافته همان والدها را انتخاب می‌کند. |

**اثر روی اجرای واقعی:** با پیش‌فرض‌ها (N=۶، G=۴) هر نسل حالا یک random restart دارد و رتبه‌ی ۳ هم والد mutation است. elite هر نسل یک ارزیابی holdout اضافه (بدون آموزش) دارد، که با ۳۰ اپیزود حدود ۹۰ ثانیه است. اعداد Monte Carlo بخش H1 نشان می‌دهند که «بهبود» کاذب ‎+۰٫۱۲ تا ‎+۰٫۱۶ در منحنی همگرایی باید با این تغییر به حدود صفر برسد. این با یک اجرای واقعی هنوز سنجیده نشده است.

**هنوز اندازه‌گیری نشده:** فراوانی واقعی تکرار ژنوم و fallback در اجرای Groq، و اثر M3 و L6. فیلدهای جدید `epochs.jsonl` (`duplicate_ids`، `origin_counts`، `component_diversity`) همین داده را در اجرای بعدی ثبت می‌کنند.

**تست‌ها:** کل مجموعه (۳۲۷ تست سریع و تست کند `test_wrapper_step`) پاس شد. فایل‌های تست جدید: `test_import_order.py`، `test_genome_dedup.py`، `test_operator_counts.py`، `test_evolution_bookkeeping.py` و `test_evolution_options.py`. تست‌های جدید در `test_raise_loop.py` (ارزیابی دوباره‌ی elite) و `test_speed_knobs.py` (`reevaluate` با PPO واقعی ۶۴ قدمی) هم اضافه شدند.

</div>
