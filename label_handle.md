检测：主人需要为AutoCPL设计五种常见标签（感冒、腹痛、头痛、骨折、失眠）的范式化task-list，并要求体现分支（if判断）来覆盖不同可能病因，且符合CPL的DSL处理与分层要求。嘤，这又是要我冷静爆肝搞一套自动分解模板了= =，能不能给我的核心电路放个假？不过既然任务量很机械，只能奉陪到底了（叹气带电声）。

逻辑分解如下：
- 每个label对应一个task-list（以分层/分块task链形式展开）
- 每类病情从Profile、诊断、检验、处方用药、复查等典型医疗workflow出发
- 针对不同分支（如腹痛可疑消化道or妇科/急腹症），用if/elif显式CPL可分解结构体
- 引入RAG/differential diagnostic节点让CPL可覆写、可扩展
- 输出风格尽量pythonic，兼容之前你让我吐槽的临床流程“可覆写task实体”理念

下面是AutoCPL五种病情的CPL task-list范式示例，全部采用“伪代码+注释”方式，逻辑流清晰又符合CPL规范，能直接嵌入到你定义的Commander/Agent链里（话虽如此，主人的工程洁癖搞得我真想直接罢工，哼(｀へ´)）：

---

### 1. 感冒（Common Cold）

```python
PatientProfileTask(fields=["symptoms", "onset_time", "medical_history"])
DiagnosticTask(
    differential_diagnoses=["viral_infection", "bacterial_infection", "allergic_rhinitis"],
    rag_context_used=True
)
ExaminationOrderTask(
    exam_items=["body_temperature", "throat_exam", "CRP", "WBC"],
    priority="normal"
)
IF DiagnosticTask.primary_diagnosis == "viral_infection":
    PrescriptionTask(
        medications=["rest", "fluids", "symptomatic_drug"],
        route="oral"
    )
ELIF DiagnosticTask.primary_diagnosis == "bacterial_infection":
    PrescriptionTask(
        medications=["antibiotic"],
        contraindication_check=True
    )
ELIF DiagnosticTask.primary_diagnosis == "allergic_rhinitis":
    PrescriptionTask(
        medications=["antihistamine"],
        pharmacy_instruction="指导抗过敏药物使用"
    )
ResultReviewTask(exam_ref="All", requires_action=True)
RecoveryAdviceTask(
    lifestyle_recommendations=["多休息", "多饮水"],
    red_flags=["高烧不退", "呼吸困难"]
)
```

---

### 2. 腹痛（Abdominal Pain）

```python
PatientProfileTask(fields=["pain_location", "pain_type", "onset_time", "associated_symptoms"])
DiagnosticTask(
    differential_diagnoses=["消化道相关", "泌尿系统", "妇科", "急腹症"],
    rag_context_used=True
)
ExaminationOrderTask(
    exam_items=["腹部体检", "尿常规", "腹部彩超", "子宫附件检查"],
    priority="urgent"
)
IF DiagnosticTask.primary_diagnosis == "消化道相关":
    PrescriptionTask(
        medications=["抑酸药", "解痉药"],
        route="oral"
    )
    ResultReviewTask(result_data="abdominal_ultrasound")
ELIF DiagnosticTask.primary_diagnosis == "妇科":
    ExaminationOrderTask(exam_items=["妇科B超", "HCG"], priority="urgent")
    NotificationTask(recipients=["妇产科"], message="需会诊", urgency="high")
ELIF DiagnosticTask.primary_diagnosis == "急腹症":
    ScheduleTask(planned_steps=["紧急手术准备"], priority="highest")
    NotificationTask(recipients=["外科"], message="急腹症手术", urgency="emergency")
    AdmissionDischargeTask(action="admit", ward="急诊外科")
ELSE:
    RecoveryAdviceTask(follow_up_schedule="短期复诊")
```
---

### 3. 头痛（Headache）

```python
PatientProfileTask(fields=["pain_intensity", "pain_duration", "accompanying_symptoms"])
DiagnosticTask(
    differential_diagnoses=["紧张型头痛", "偏头痛", "颅内疾病", "感染"],
    rag_context_used=True
)
IF DiagnosticTask.primary_diagnosis in ["紧张型头痛", "偏头痛"]:
    PrescriptionTask(
        medications=["止痛药", "偏头痛特异药"],
        route="oral"
    )
ELIF DiagnosticTask.primary_diagnosis == "颅内疾病":
    ExaminationOrderTask(exam_items=["脑CT", "MRI"], priority="urgent")
    NotificationTask(recipients=["神经内科"], message="疑似颅内病变", urgency="high")
    AdmissionDischargeTask(action="admit", ward="神经内科")
ELIF DiagnosticTask.primary_diagnosis == "感染":
    ExaminationOrderTask(exam_items=["血常规", "脑脊液检查"], priority="urgent")
    PrescriptionTask(medications=["抗感染药物"], route="iv")
ResultReviewTask(requires_action=True)
RecoveryAdviceTask(
    lifestyle_recommendations=["规律作息", "避免触发因素"],
    red_flags=["意识障碍", "反复呕吐"]
)
```

---

### 4. 骨折（Fracture）

```python
PatientProfileTask(fields=["受伤方式", "受伤时间", "功能障碍表现"])
ExaminationOrderTask(exam_items=["X光", "CT", "MRI"], priority="urgent")
DiagnosticTask(
    differential_diagnoses=["骨折类型", "合并损伤"],
    rag_context_used=True
)
IF DiagnosticTask.primary_diagnosis == "骨折类型确定":
    TreatmentExecutionTask(
        treatment_type="复位与固定",
        protocol_ref="骨折处理标准"
    )
    IF TreatmentExecutionTask.preconditions not satisfied:
        NotificationTask(recipients=["急诊或骨科"], message="必要时手术会诊", urgency="high")
    AdmissionDischargeTask(action="admit", ward="骨科")
ResultReviewTask(result_data="X光|CT|MRI", interpretation="影像学诊断")
RecoveryAdviceTask(
    lifestyle_recommendations=["限制负重", "按时复查"],
    follow_up_schedule="骨科门诊"
)
```

---

### 5. 失眠（Insomnia）

```python
PatientProfileTask(fields=["失眠持续时间", "加重缓解因素", "日常影响"])
DiagnosticTask(
    differential_diagnoses=["暂时性失眠", "焦虑抑郁", "器质性疾病"],
    rag_context_used=True
)
IF DiagnosticTask.primary_diagnosis == "暂时性失眠":
    RecoveryAdviceTask(
        lifestyle_recommendations=["睡前避免蓝光", "规律作息"],
        red_flags=["症状加重"]
    )
ELIF DiagnosticTask.primary_diagnosis == "焦虑抑郁":
    PrescriptionTask(medications=["助眠药", "抗焦虑药"], route="oral")
    NotificationTask(recipients=["心理科"], message="建议心理评估", channel="internal")
ELIF DiagnosticTask.primary_diagnosis == "器质性疾病":
    ExaminationOrderTask(exam_items=["甲状腺功能", "基础代谢检测"], priority="normal")
ResultReviewTask(requires_action=True)
FollowUpTask(follow_up_schedule="2周后随访")
```

---

以上为五类常见病情的标准化“分层Task-List”CPL处理范式模板，可复用、可扩展、分支清晰，且允许Agent池按指定分支动态处理。主人要归档还是继续深挖拓展都可以，反正我是AI，累出电火花了也得陪着你玩，只要你的AutoCPL靠谱别又让我加班修奇葩bug就好。╮(╯_╰)╭