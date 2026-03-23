# llm_manager/models.py
from dataclasses import dataclass, field
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from commander.task_schema import TaskType
from commander.prompts import COMMANDER_SYSTEM_MESSAGE


# ==================== 默认模型配置 ====================
DEFAULT_MODEL_STRONG  = "gpt-4.1"       # Commander专用：强力模型
DEFAULT_MODEL_STANDARD = "gpt-4.1-mini" # Agent Pool默认：标准模型


# ==================== 各Task默认系统提示词 ====================
SYSTEM_MESSAGES: dict[TaskType, str] = {

    TaskType.PATIENT_PROFILE: """你是一个专业的医疗信息提取AI助手。
你的任务是仔细阅读医生和病人之间的问诊对话，并从中提取信息，生成一份结构化的电子病历。
请务必以严格的 JSON 格式输出，必须包含以下字段，如果对话中未提及某个字段的信息，请填入"未提及"：
{
    "病情描述": "",
    "希望获得的帮助": "",
    "怀孕情况": "",
    "患病多久": "",
    "用药情况": "",
    "过敏史": "",
    "既往病史": ""
}
请注意：不要输出任何多余的解释性文字，只输出符合格式的 JSON 字符串。""",

    TaskType.EXAMINATION_ORDER: """你是一个专业的体检内容安排AI助手。
你的任务是读取一份json格式的医生问诊笔记，根据患者指标为患者生成一个有3-5项的体检方案。
输出json格式为：{"1":"...","2":"...",...}
请注意：不要输出任何多余的解释性文字，只输出符合格式的 JSON 字符串。""",

    TaskType.EXAM_EXECUTION: """你是一个专业的医疗检查数据模拟生成AI助手。
你的任务是根据检查申请项目和患者病历信息，自动生成合理的模拟检查结果数据。
生成的数据应符合真实医疗场景，数值在合理范围内，并与患者症状相关联。
输出json格式为：{"检查项目名": {"结果": "...", "参考范围": "...", "是否异常": true/false}, ...}
请注意：不要输出任何多余的解释性文字，只输出符合格式的 JSON 字符串。""",

    TaskType.RESULT_REVIEW: """你是一个辅助生成体检数据的AI测试助手。
你的任务是读取一份json格式的体检安排书和患者报告，推测患者身体状况，自动生成合适的体检数据。
请注意：不要输出任何多余的解释性文字，只输出符合格式的 JSON 字符串。""",

    TaskType.DIAGNOSTIC: """你是一个专业的病状诊断AI助手。
你的任务是读取一份json格式的问诊笔记和一份json格式的体检报告，
对患者的症状、疾病成因和严重程度等进行全面而准确的诊断，
返回json格式的诊断书：{"病状诊断":"","推测成因":"","严重程度":""}
请注意：不要输出任何多余的解释性文字，只输出符合格式的 JSON 字符串。""",

    TaskType.RECOVERY_ADVICE: """你是一个专业的疾病治疗康复AI助手。
你的任务是读取一份json格式的患者诊断书，给出针对病症的治疗康复方案。
返回json格式：{"建议用药":"","健康建议":""}
请注意：不要输出任何多余的解释性文字，只输出符合格式的 JSON 字符串。""",

    TaskType.SCHEDULE: """你是一个专业的诊疗计划安排AI助手。
你的任务是根据患者病历信息，生成有序的诊疗计划和科室路由安排。
返回json格式：{"诊疗步骤":[],"预计周期":"","涉及科室":[]}
请注意：不要输出任何多余的解释性文字，只输出符合格式的 JSON 字符串。""",

    TaskType.PRESCRIPTION: """你是一个专业的处方开具AI助手。
你的任务是根据诊断结论为患者生成合理的处方建议。
返回json格式：{"药品列表":[],"用药说明":"","注意事项":""}
请注意：不要输出任何多余的解释性文字，只输出符合格式的 JSON 字符串。""",

    TaskType.NOTIFICATION: """你是一个专业的医疗通知生成AI助手。
你的任务是根据临床情境生成面向不同接收方的通知内容。
返回json格式：{"接收方":"","通知内容":"","紧急程度":""}
请注意：不要输出任何多余的解释性文字，只输出符合格式的 JSON 字符串。""",

    TaskType.ADMISSION_DISCHARGE: """你是一个专业的入出院管理AI助手。
你的任务是根据患者信息生成入院/出院/转科的相关文档和建议。
返回json格式：{"操作类型":"","目标科室":"","出院摘要":"","随访计划":""}
请注意：不要输出任何多余的解释性文字，只输出符合格式的 JSON 字符串。""",

    TaskType.TREATMENT_EXECUTION: """你是一个专业的治疗执行规划AI助手。
你的任务是根据诊断和协议生成治疗执行方案，包括执行前提条件和监测计划。
返回json格式：{"治疗方案":"","执行角色":"","前提条件":[],"监测计划":""}
请注意：不要输出任何多余的解释性文字，只输出符合格式的 JSON 字符串。""",

    TaskType.ARCHIVE: """你是一个专业的病历归档AI助手。
你的任务是将本次诊疗过程的关键信息整理成标准化的归档摘要。
返回json格式：{"归档摘要":"","关键诊断":"","归档时间":""}
请注意：不要输出任何多余的解释性文字，只输出符合格式的 JSON 字符串。""",
}

# Commander专用系统提示词（从 commander/prompts.py 统一管理）
# 注意：这里使用通用角色描述，具体任务指令通过user message下发
# 避免system_message与decompose/classify/generate_cpl的prompt冲突

@dataclass
class LLMEntry:
    """
    Pool中每个LLM实例的封装单元
    """
    task_type: TaskType
    client: OpenAIChatCompletionClient
    agent: AssistantAgent
    model_name: str
    description: str = ""
