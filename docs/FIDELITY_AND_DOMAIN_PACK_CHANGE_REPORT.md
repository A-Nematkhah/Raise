# گزارش کامل تغییرات — Domain Pack و سخت‌سازی Fidelity پایهٔ EvoNav

**مخاطب:** خود پروژه / ناظر پایان‌نامه  
**بازهٔ کار:** سپتامبر ۲۰۲۶ (پس از قفل Score1؛ شامل Domain Pack + fidelity + پیش‌فرض‌های مقاله)  
**نقطهٔ ذخیره / تگ baseline:** `baseline-pre-amfrs` روی ریموت `baseline`  
**مقالهٔ مرجع baseline:** EvoNav (arXiv:2605.11859) — فایل محلی `Evonav.pdf`  
**هدف این گزارش:** توضیح ساده، کامل و دقیقِ اینکه **قبل چه بود**، **چه کردیم**، و **بعد چه شد** — بدون اغراق «بازتولید کامل مقاله».

---

## ۱) خلاصهٔ یک‌صفحه‌ای

پروژه یک **پیاده‌سازی قابل‌اجرای Algorithm 1 مقالهٔ EvoNav** روی شبیه‌ساز CrowdNav++ بود. اسکلت درست بود، ولی:

1. بعضی ادعاهای «وفادار به مقاله / byte-faithful» بیش از حد قوی بودند؛
2. چند فاصلهٔ واقعی با متن مقاله (رتبه‌بندی، واحد K2، Score1، دیتاست، elitism) در جدول انحراف‌ها شفاف نبود؛
3. الگوریتم جستجوی reward از محیط شبیه‌سازی به‌هم چسبیده بود و اضافه کردن محیط دوم سخت می‌شد.

در این دوره دو مسیر موازی انجام شد:

| مسیر | هدف |
|------|-----|
| **A. Domain Pack** | جدا کردن «موتور الگوریتم» از «محیط CrowdNav» بدون شکستن نتیجهٔ پیش‌فرض |
| **B. Fidelity hardening** | صادق‌کردن مستندات + اصلاح باگ‌های واقعی + فلگ برای نزدیک‌شدن به مقاله |

**نتیجهٔ کلی:**  
پایهٔ فعلی برای مقایسهٔ بعدی با AMFRS **صادق‌تر و ماژولارتر** است. هنوز ادعای «بازتولید عددی Table 1 مقاله» نداریم مگر با run مقیاس‌کاغذی و دیتاست تازه‌جمع‌شده.

---

## ۲) وضعیت قبل از این تغییرات

### ۲.۱ چه چیزی درست کار می‌کرد؟

- مسیر end-to-end: LLM → sandbox → Stage I (Score1) → Stage II (A2C) → Stage III (PPO)
- اعداد ساختاری جدول‌های ۳–۶ تا حد زیادی در کد بود: `N=8`, `G1=10`, `G2=16`, `E2=50`, `T_short=100`, `G3=3`, `E3=500`, H-sweep
- انتقال همهٔ N کاندید بین stageها (مثل Algorithm 1، نه «فقط زیرمجموعه»)
- تست‌ها و قفل Score1 برای padding / degeneracy (commit `f8d3619`)

### ۲.۲ مشکلات اصلی (قبل)

| موضوع | مشکل |
|--------|------|
| مستند baseline | برچسب «byte-faithful» شامل چیزهایی مثل elitism بود که در Algorithm 1 نیست |
| رتبه‌بندی R2/R3 | مقاله: ارزیابی چندهدفه با LLM؛ کد: اسکالر ثابت `SR − CR − 0.5·TR` |
| واحد K2 | مقاله §4.3.2: **gradient steps**؛ کد: **env steps** (با K2=8000 آپدیت A2C خیلی کم می‌شد) |
| Score1 / Figure 3 | با `nav_length = f+1` ترجیح Success کوتاه بر بلند practically خاموش بود |
| دیتاست Stage I | ۳×ORCA و ۳×SF deterministic تکراری → حدود ۵ traj یکتا از ۱۰ |
| Mutation prompt | کد روی نیمه‌ی ضعیف mutate می‌کرد؛ پرامپت می‌گفت «elite / high-performing» |
| Reflection | هر نسل overwrite می‌شد؛ مقاله از انباشت یادداشت‌ها حرف می‌زند |
| معماری | pipeline مستقیم CrowdNav/Stage trainers را import می‌کرد؛ محیط pluggable نبود |

### ۲.۳ واقعیت اجراهای قبلی

اجرای مستند `run_scaled_h5_gst` با بودجهٔ کمتر از مقاله بود و SR≈0.65 زیر ORCA/GST. یعنی قبل از این کار هم «پیاده‌سازی» داشتیم، نه «بازتولید ادعای Table 1».

---

## ۳) مسیر کار (ترتیب منطقی)

```text
1) Domain Pack (جداسازی محیط بدون تغییر رفتار پیش‌فرض)
2) بازبینی مقاله + گزارش fidelity خارجی
3) فاز A: مستند صادق (BASELINE_REPORT)
4) فاز B: اصلاح Score1 + collector + mutation/reflection
5) فاز C: فلگ‌های K2 / final-rank / elitism
6) فاز D: لاگ Spearman بین stageها (proxy consistency)
```

---

## ۴) بخش اول — Domain Pack (جداسازی محیط)

### ۴.۱ ایده به زبان ساده

مثل این که قبلاً «آشپزخانه و رستوران یکی» بودند.  
حالا می‌گوییم:

- **موتور الگوریتم** فقط با یک قرارداد حرف می‌زند؛
- هر محیط داخل یک پوشهٔ **Domain Pack** ملزوماتش را می‌دهد؛
- پیش‌فرض همچنان **crowdnav** است تا baseline عددی نشکند.

### ۴.۲ ساختار جدید

```text
evonav_env/crowd_nav/domains/
  README.md                 ← راهنمای افزودن محیط واقعی (بدون stub جعلی)
  base.py                   ← DomainPack + EnvAdapter
  __init__.py               ← load_domain / make_*_for_domain
  crowdnav/
    pack.py                 ← get_pack()
    prompts.py              ← منبع اصلی Appendix D
    stage1.py               ← Score1 / smoke
    adapter.py              ← پل Stage II/III به trainerهای قبلی
    spec.md                 ← توضیح تسک / state / متریک
```

### ۴.۳ قرارداد هر بستهٔ دامنه

هر محیط باید این‌ها را بدهد:

1. **spec** — هدف تسک، state، اصول reward، متریک‌ها  
2. **prompts + seed reward** — برای LLM  
3. **Stage I** — `make_score_fn` (برای CrowdNav همان Score1)  
4. **Stage II/III** — `train_and_eval` از طریق adapter  
5. ثبت در registry (`load_domain("crowdnav")`)

### ۴.۴ اتصال به pipeline

- CLI: `--domain crowdnav` (پیش‌فرض)
- `reward_search/prompts.py` فقط **re-export** از pack است (متن prompt CrowdNav ثابت ماند)
- Stage I از `make_score_fn_for_domain`
- Stage II/III از `make_stage2/3_trainer_for_domain` (حتی stub از همین مسیر)

### ۴.۵ چیزی که عمداً انجام نشد

- دامنهٔ دوم واقعی ساخته نشد (طبق درخواست: بدون stub)
- `RewardState` هنوز crowd-centric است (عمومی‌سازی فقط وقتی محیط دوم واقعی بیاید)

---

## ۵) بخش دوم — سخت‌سازی Fidelity نسبت به مقاله

منبع اصلی فاصله‌ها: متن مقاله (Algorithm 1، §4.3، Figure 3، Appendix D) + بازبینی کد/گزارش خارجی + تأیید روی دیتاست محلی.

### ۵.۱ فاز A — صداقت مستند

**فایل اصلی:** `BASELINE_REPORT.md`

تغییر مفهومی:

- سطل «Byte-faithful» → «Faithful structural» فقط برای چیزهای واقعاً جور با Alg 1 / جداول
- جدول **Documented deviations** گسترش یافت: R2/R3، واحد K2، elitism، split نسل، seed، interface، N_traj، …
- `selection.py` صریحاً می‌گوید اسکالر ≠ رتبه‌بندی LLM مقاله

**اثر:** مقایسه‌های بعدی «AMFRS در برابر EvoNav» دیگر روی برچسب غلط بنا نمی‌شود.

### ۵.۲ فاز B1 — تنوع دیتاست Stage I

**فایل:** `scripts/collect_stage1_dataset.py`

| قبل | بعد |
|-----|-----|
| ۳ ORCA تمیز + ۳ SF تمیز (deterministic تکراری) | ۱ ORCA + ۱ SF تمیز + نویزهای مدرج + ۲ random |
| seed یکسان برای همهٔ traj یک scenario | RNG جدا per-trajectory |
| کامنت «Paper uses 3 ORCA+…» | صریح: mix مال ماست؛ مقاله فقط N_traj=10 diverse گفته |

**شواهد:**

- دیتاست قدیمی: ~۵ unique از ۱۰  
- نمونهٔ جدید `data/_fidelity_stage1_sample` (M=3): **۹ unique از ۱۰**

**توجه:** دیتاست اصلی `data/stage1_dataset` باید با دستور زیر دوباره جمع شود تا اثر کامل دیده شود:

```bash
cd evonav_env
python scripts/collect_stage1_dataset.py --out data/stage1_dataset
```

### ۵.۳ فاز B2 — Score1 و Figure 3

**فایل:** `reward_search/scoring.py`

| قبل | بعد |
|-----|-----|
| `nav_length = f + 1` برای همه | `nav_length = min(f + 1, traj.length)` |

**معنی:**  
تا وقتی traj هنوز تمام نشده، طول آینده لو نمی‌رود؛ بعد از پایان Success کوتاه، در فریم‌های بعدی همان طول کوتاه می‌ماند و ترجیح «Success کوتاه ≻ Success بلند» زنده می‌شود.

تست جدید: دو reward که فقط در ترجیح short/long فرق دارند باید Score1 متفاوت بگیرند.

### ۵.۴ فاز B3 — Mutation و Reflection

**فایل‌ها:** `domains/crowdnav/prompts.py`, `evolver.py`

| موضوع | قبل | بعد |
|--------|-----|-----|
| پرامپت mutation | «high-performing / elite» | «underperforming parent» هم‌تراز §4.2 |
| Reflection | overwrite هر نسل (best/worst) | انباشت تا ۳ نسل + امتیاز همهٔ کاندیدها |

کد همچنان روی **نیمه‌ی پایین** جمعیت mutate می‌کند؛ الان پرامپت با همین رفتار جور است.

### ۵.۵ فاز C — پیش‌فرض‌های وفادار به مقاله

پیش‌فرض pipeline/CLI اکنون همان تنظیمات مقاله است (نه فلگ اختیاری).

#### C1 — واحد K2

```text
--k2-unit gradient_steps     ← پیش‌فرض (مقاله §4.3.2)
--k2-unit env_steps          ← بودجهٔ مهندسی اختیاری
--stage2-train-steps 8000    ← پیش‌فرض Table 5
```

با `gradient_steps` و K2=8000:

`env_steps = 8000 × num_steps × num_processes`  
تا تعداد آپدیت A2C تقریباً برابر ۸۰۰۰ شود.

#### C2 — رتبه‌بندی نهایی و elitism

```text
--final-rank llm             ← پیش‌فرض Alg.1 R2/R3 (seed → lex fallback)
--final-rank scalar          ← SR−CR−0.5·TR مهندسی
--elitism                    ← روشن کردن inject/protect (غیرمقاله؛ پیش‌فرض off)
```

خروجی: `final_ranking_R2` / `final_ranking_R3` داخل JSONهای stage و `manifest`.  
`best_stage2.json` / `best_stage3.json` از همان رتبهٔ R2/R3 انتخاب می‌شوند.

اگر LLM واقعی نباشد (مثلاً `--llm seed`)، برای mode=llm یک **رتبهٔ lexicographic چندمتریکه** به‌عنوان fallback استفاده می‌شود (نه اسکالر ساده) و در artifact ثبت می‌شود.

**اجرای پیش‌فرض نزدیک به مقاله:**

```bash
python scripts/run_evonav.py \
  --llm groq --device cuda ...
# K2=8000 gradient_steps, final_rank=llm, elitism=off already default
```

### ۵.۶ فاز D — Proxy consistency (§4.3.4)

**فایل جدید:** `reward_search/proxy_consistency.py`

هر run کامل می‌نویسد:

- `proxy_consistency.json`
- خلاصه در `manifest.json`

معیارها (روی fingerprint کد reward، چون id بعد از refine عوض می‌شود):

- Spearman ρ بین Stage I↔II و II↔III  
- top-k preservation

این همان چیزی است که مقاله برای «آیا proxy رتبه را حفظ می‌کند؟» ادعا می‌کند و قبلاً در ریپو لاگ سیستماتیک نداشت.

---

## ۶) فایل‌ها و نقاط ورود مهم

| مسیر | نقش |
|------|-----|
| `BASELINE_REPORT.md` | منبع حقیقت انحراف‌ها / وفاداری |
| `crowd_nav/domains/` | Domain Pack |
| `reward_search/pipeline.py` | orchestration + فلگ‌ها + consistency |
| `reward_search/scoring.py` | Score1 / Figure 3 |
| `reward_search/ranking.py` | R2/R3 |
| `reward_search/proxy_consistency.py` | Spearman بین stageها |
| `scripts/collect_stage1_dataset.py` | دیتاست Stage I |
| `scripts/run_evonav.py` | CLI فلگ‌های جدید |
| `domains/README.md` | راهنمای افزودن محیط جدید |

تست‌های اضافه‌شده/به‌روز:

- `test_domain_pack_baseline.py`
- `test_stage1_collector_schedule.py`
- `test_score1_baseline_lock.py` (Success short/long)
- `test_fidelity_flags.py`
- `test_proxy_consistency.py`

در لحظهٔ ثبت این گزارش: suite غیرslow حدود **۱۳۸** تست سبز بود.

---

## ۷) قبل / بعد — جدول جمع‌بندی

| محور | قبل | بعد |
|------|-----|-----|
| معماری | الگوریتم ⟷ CrowdNav چسبیده | Domain Pack؛ پیش‌فرض crowdnav |
| ادعای مستند | byte-faithful گمراه‌کننده | جدول deviation صادق |
| Score1 Figure 3 | Success short/long خاموش | `min(f+1, len)` |
| دیتاست | ~۵/۱۰ unique | schedule متنوع؛ نمونه ۹/۱۰ |
| Mutation prompt | ناهماهنگ با کد | underperforming parent |
| Reflection | تک‌نسلی | انباشتی (حداکثر ۳) |
| K2 | فقط env-step | پیش‌فرض **gradient_steps** + 8000 |
| R2/R3 | فقط اسکالر پنهان | پیش‌فرض **llm** (+ انتخاب best از رتبه) |
| Elitism | همیشه روشن و کم‌اعلام | پیش‌فرض **off**؛ `--elitism` اختیاری |
| Proxy consistency | نبود | `proxy_consistency.json` |

---

## ۸) چه چیزی هنوز «انجام‌نشده / خارج از این گزارش» است؟

این‌ها عمداً یا هنوز باز مانده‌اند:

1. **بازتولید Table 1 مقاله** (K3=1e7، چند seed، LLM قوی) — نیاز به run علمی جدا  
2. **Recollect کامل** `data/stage1_dataset` با M=100 (نمونهٔ کوچک زده شد)  
3. **امضای دقیق مقاله** (`inst/traj` یا `cal_reward(st)`) — مقاله خودش ناسازگار است؛ قرارداد فعلی sandbox حفظ شد  
4. **کپی تحت‌اللفظی seed باگ‌دار D.5** مقاله — عمداً potential تا goal نگه داشته شد  
5. **AMFRS** (Pareto / archive / multi-objective evolution پایان‌نامه) — هنوز شروع نشده؛ این کار فقط پایه را تمیز کرد  
6. **دامنهٔ شبیه‌سازی دوم** — فقط قرارداد و مستند آماده است

ناسازگاری‌های داخلی خود مقاله (مثلاً متن «زیرمجموعه به Stage III» در برابر Algorithm 1 که همهٔ N را می‌فرستد؛ یا هزینهٔ GPU در Table 9) را دنبال نکردیم؛ Algorithm 1 را مرجع گرفتیم.

---

## ۹) توصیهٔ عملی برای ادامه

### اگر هدف «baseline صادق برای AMFRS» است

1. دیتاست Stage I را کامل recollect کنید.  
2. یک run مرجع با فلگ‌های paper-like بگیرید و `proxy_consistency.json` را نگه دارید.  
3. هر feature AMFRS را روی همین snapshot diff کنید.

### اگر هدف «نزدیک‌تر شدن به عدد مقاله» است

علاوه بر بالا: `--k2-unit gradient_steps`، بودجهٔ Stage III نزدیک paper_scale، LLM واقعی، و ارزیابی در برابر ORCA/GST/CrowdNav++ با پروتکل یکسان.

---

## ۱۰) جمع‌بندی نهایی

این مجموعه تغییرات دو کار کرد:

1. **مهندسی:** الگوریتم را از محیط جدا کرد تا بعداً محیط دیگر / AMFRS تمیز اضافه شود.  
2. **علمی/وفاداری:** فاصله با مقاله را صادقانه نوشت، چند باگ واقعی را بست، و برای بقیه مسیر فلگ گذاشت — بدون شکستن پیش‌فرض عملی قبلی.

وضعیت درست برای گفتن به ناظر:

> «پایهٔ EvoNav Algorithm 1 اکنون ماژولار و از نظر fidelity مستند و اصلاح‌شده است؛ هنوز ادعای بازتولید کامل نتایج Table 1 مقاله را مطرح نمی‌کنیم تا run مقیاس‌کاغذی با دیتاست تازه انجام شود.»

---

*پایان گزارش.*
