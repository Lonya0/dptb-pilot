## 1.0.1 update
+ 在`dptb_pilot/tools/modules/deeptb/__init__.py`中增添了声明`remotely_tools`
+
+ 将原有的`Local`模式和`Bohr`模式更改为`Local`模式和`Remote`模式
当进入`Remote`模式后，前端会显示一个远程机器列表。
在列表的右上角有一个加号，可以点击添加一个新的远程机器。
对于一个新的机器，用户可以对其进行命名，并且选择类型为`Bohrium`或`Slurm`。
在用户尝试使用在`remotely_tools`中的工具时，会在弹出的参数确认列表中要求用户选择使用哪个远程机器进行运行，默认自动选第一个机器。
对于`Bohrium`类型，还有几个输入项：`username`,`password`,`project_id`,`image_name`,`scass_type`
前三个是必填选项，对应用户的Bohrium用户名、密码和项目ID，后两项为可选选项。
在传递参数时如往常一样修改Executor为：
```Python
{
  type: 'dispatcher',
  machine: {
    batch_type: 'Bohrium',
    context_type: 'Bohrium',
    remote_profile: {
      email: Config.username,
      password: Config.password,
      program_id: parseInt(Config.project_id),
      input_data: {
        image_name: Config.image_name if Config.image_name else 'registry.dp.tech/dptech/dp/native/prod-19853/dpa-mcp:0.0.0',
        job_type: 'container',
        platform: 'ali',
        scass_type: Config.scass_type if Config.scass_type else '1 * NVIDIA V100_32g'
      }
    }
  }
}
```
修改Storage为：
```Python
{
  type: 'bohrium',
  username: Config.username,
  password: Config.password,
  program_id: parseInt(Config.project_id)
}
```
对于`Slurm`类型，还有几个输入项：`remote_root`,`hostname`,`username`,`key_filename`,`number_node`,`gpu_per_node`,`cpu_per_node`,`queue_name`,`custom_flags`
其中`remote_root`,`hostname`,`username`,`key_filename`,`number_node`,`queue_name`，其余为可选选项。
在传递参数时修改Executor为：
```Python
{
  type: 'dispatcher',
  machine: {
    batch_type: "Slurm",
    context_type: "SSHContext",
    local_root: "./",
    remote_root: Config.remote_root,
    remote_profile: { 
        hostname: Config.hostname,
        username: Config.username,
        timeout: 600,
        port: 22,
        key_filename: Config.key_filename 
    }
  },
  resource:{
    number_node: Config.get("number_node", 1),
    gpu_per_node: Config.get("gpu_per_node", 0),
    cpu_per_node: Config.get("cpu_per_node", 1),
    queue_name: Config.queue_name,
    custom_flags: [Config.get("custom_flags"), ""],
    source_list: [],
    module_list: []
  }
}
```

### 1.0.1 fix-a
+ 在每次传递executor和storage参数时，都在日志中打印出来，有可能executor传递的完全不正确。在上次运行中，一个任务直接将我的服务器卡炸了，我认为其运行在了服务器本地而不是远程，因为远程也没有收到文件，请修复该问题。
+ 在每个远程机器面板上，添加一个修改按钮，点击可以修改参数
+ 在远程机器的输入环节，每个中文后都用括号标出对应变量名，如：“用户名(username)”

### 1.0.1 fix-b
+ 依然有上个问题，一旦尝试使用提交到Slurm任务时，会直接报错并崩溃，此时前端和mcp-tools-server(dptb-tools)都会崩溃，请阅读log1.txt中的日志，并解决崩溃问题

### 1.0.1 fix-c
+ 我手动修复了问题，问题在与之前启动dptb-tools时，dptb_pilot/tools/init.py位置的mcp使用的是fastmcp而不是dp，现在调取参数时就会正常要求Executor和Storage了
+ 但是也发现了问题，该框架的设计上是跳过智能体对Executor和Storage的调用，需要通过api服务器自动选择机器，目前看来还没有实现
+ 我需要你实现确保Executor的传参和Storage的传参是自动根据远程机器的数据自动生成的
+ 同时，对用户展示时，隐藏这两个参数，即不给用户展示这两个参数，不允许手动修改，而是自动传参，传参格式按上方格式
+ 此外，我需要你添加一个“终止执行”按键在“提交修改”按键右侧占据一半的位置，用来终止智能体的执行，并避免让其继续尝试提交计算任务

### 1.0.1 fix-d
+ 终止执行完全没用，点击没效果
+ 隐藏参数也没实现
+ Executor和Storage提交也有问题，以下是后台信息，请你查看哪一步有问题：
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] ================================================================================
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [ModifyParams] 收到参数修改请求
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [ModifyParams] Session ID: jPAKkXkMuXfC5oyf3hwPKOW1N4xHa1WV
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [ModifyParams] 工具名称: band_with_baseline_model
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [ModifyParams] 执行模式: Remote
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [ModifyParams] 选中的机器ID: 1768480151460
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [ModifyParams] 修改前的Schema: {
  "name": "band_with_baseline_model",
  "description": "\n    使用基准模型预测结构能带。\n\n    参数:\n        basemodel: 使用的baseline model类型。可选：poly4，poly2\n        structure_file_name: 输入的结构文件路径。结构文件 
应为vasp的格式\n        work_path: 能带信息的保存路径。注意应该是文件夹而不是文件。\n\n    返回:\n        包含能带文件路径的字典。\n\n    抛出:\n        AssumptionError: 某些数据输入不合规\n        RuntimeError: 写入配置文件失败。\n    ",
  "input_schema": {
    "properties": {
      "basemodel": {
        "default": "poly4",
        "title": "Basemodel",
        "type": "string",
        "agent_input": "poly4",
        "user_input": "poly4"
      },
      "structure_file_name": {
        "default": "your_structure_file_name",
        "title": "Structure File Name",
        "type": "string",
        "agent_input": "7_0.vasp",
        "user_input": "7_0.vasp"
      },
      "work_path": {
        "default": ".",
        "title": "Work Path",
        "type": "string",
        "agent_input": "C:\\Users\\lonya\\PycharmProjects\\dptb-pilot\\workspace\\jPAKkXkMuXfC5oyf3hwPKOW1N4xHa1WV\\files",
        "user_input": "C:\\Users\\lonya\\PycharmProjects\\dptb-pilot\\workspace\\jPAKkXkMuXfC5oyf3hwPKOW1N4xHa1WV\\files"
      },
      "executor": {
        "anyOf": [
          {
            "additionalProperties": true,
            "type": "object"
          },
          {
            "type": "null"
          }
        ],
        "default": null,
        "title": "Executor",
        "agent_input": {},
        "user_input": {}
      },
      "storage": {
        "anyOf": [
          {
            "additionalProperties": true,
            "type": "object"
          },
          {
            "type": "null"
          }
        ],
        "default": null,
        "title": "Storage",
        "agent_input": {},
        "user_input": {}
      }
    }
  },
  "parameters": {}
}
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] ================================================================================
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] ================================================================================
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [AutoGenerate] 开始自动生成 Executor 和 Storage 参数
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [AutoGenerate] 机器类型: Slurm
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [AutoGenerate] 机器配置: {
  "remote_root": "/public4/home/a0s000985/yyt/CNT/press_band_dptb",
  "hostname": "ssh.cn-zhongwei-1.paracloud.com",
  "username": "a0s000985@BSCC-A",
  "key_filename": "/root/.ssh/id_ed25519",
  "number_node": "1",
  "gpu_per_node": "0",
  "cpu_per_node": "64",
  "queue_name": "amd_256",
  "custom_flags": "source /public4/home/a0s000985/yyt/envs/negf.sh"
}
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [AutoGenerate] 完成 Executor 和 Storage 参数自动生成
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] ================================================================================
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] ================================================================================
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [ModifyParams] 生成 Executor 和 Storage 后的Schema: {
  "name": "band_with_baseline_model",
  "description": "\n    使用基准模型预测结构能带。\n\n    参数:\n        basemodel: 使用的baseline model类型。可选：poly4，poly2\n        structure_file_name: 输入的结构文件路径。结构文件 
应为vasp的格式\n        work_path: 能带信息的保存路径。注意应该是文件夹而不是文件。\n\n    返回:\n        包含能带文件路径的字典。\n\n    抛出:\n        AssumptionError: 某些数据输入不合规\n        RuntimeError: 写入配置文件失败。\n    ",
  "input_schema": {
    "properties": {
      "basemodel": {
        "default": "poly4",
        "title": "Basemodel",
        "type": "string",
        "agent_input": "poly4",
        "user_input": "poly4"
      },
      "structure_file_name": {
        "default": "your_structure_file_name",
        "title": "Structure File Name",
        "type": "string",
        "agent_input": "7_0.vasp",
        "user_input": "7_0.vasp"
      },
      "work_path": {
        "default": ".",
        "title": "Work Path",
        "type": "string",
        "agent_input": "C:\\Users\\lonya\\PycharmProjects\\dptb-pilot\\workspace\\jPAKkXkMuXfC5oyf3hwPKOW1N4xHa1WV\\files",
        "user_input": "C:\\Users\\lonya\\PycharmProjects\\dptb-pilot\\workspace\\jPAKkXkMuXfC5oyf3hwPKOW1N4xHa1WV\\files"
      },
      "executor": {
        "anyOf": [
          {
            "additionalProperties": true,
            "type": "object"
          },
          {
            "type": "null"
          }
        ],
        "default": null,
        "title": "Executor",
        "agent_input": {},
        "user_input": {}
      },
      "storage": {
        "anyOf": [
          {
            "additionalProperties": true,
            "type": "object"
          },
          {
            "type": "null"
          }
        ],
        "default": null,
        "title": "Storage",
        "agent_input": {},
        "user_input": {}
      }
    }
  },
  "parameters": {}
}
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] ================================================================================
[2026-01-22 19:47:21] [INFO] [dptb_pilot.core.guardrail] ================================================================================
[2026-01-22 19:47:21] [INFO] [dptb_pilot.core.guardrail] [ExtractArguments] 开始提取参数
[2026-01-22 19:47:21] [INFO] [dptb_pilot.core.guardrail] [ExtractArguments] 工具名称: band_with_baseline_model
[2026-01-22 19:47:21] [INFO] [dptb_pilot.core.guardrail] [ExtractArguments] 最终参数: {'basemodel': 'poly4', 'structure_file_name': '7_0.vasp', 'work_path': 'C:\\Users\\lonya\\PycharmProjects\\dptb-pilot\\workspace\\jPAKkXkMuXfC5oyf3hwPKOW1N4xHa1WV\\files', 'executor': {}, 'storage': {}}
[2026-01-22 19:47:21] [INFO] [dptb_pilot.core.guardrail] ================================================================================
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [ModifyParams] 提取后的参数: {'basemodel': 'poly4', 'structure_file_name': '7_0.vasp', 'work_path': 'C:\\Users\\lonya\\PycharmProjects\\dptb-pilot\\workspace\\jPAKkXkMuXfC5oyf3hwPKOW1N4xHa1WV\\files', 'executor': {}, 'storage': {}}
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [ModifyParams] Executor 参数: None
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [ModifyParams] Storage 参数: None
[2026-01-22 19:47:21] [INFO] [dptb_pilot.server.app] [ModifyParams] 触发事件，恢复 agent 执行

## 1.0.2 update
+ 之前的问题依旧没有修复，但是Executor和Storage确实被隐藏了，但是两者最后还是空值，以下是dptb-pilot的log：
```text
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] ================================================================================
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [ModifyParams] 收到参数修改请求
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [ModifyParams] Session ID: jPAKkXkMuXfC5oyf3hwPKOW1N4xHa1WV
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [ModifyParams] 工具名称: band_with_baseline_model
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [ModifyParams] 执行模式: Remote
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [ModifyParams] 选中的机器ID: 1768480151460
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [ModifyParams] 修改前的Schema: {
  "name": "band_with_baseline_model",
  "description": "\n    使用基准模型预测结构能带。\n\n    参数:\n        basemodel: 使用的baseline model类型。可选：poly4，poly2\n        structure_file_name: 输入的结构文件路径。结 
构文件应为vasp的格式\n        work_path: 能带信息的保存路径。注意应该是文件夹而不是文件。\n\n    返回:\n        包含能带文件路径的字典。\n\n    抛出:\n        AssumptionError: 某些数输入不合规。\n        RuntimeError: 写入配置文件失败。\n    ",
  "input_schema": {
    "properties": {
      "basemodel": {
        "default": "poly4",
        "title": "Basemodel",
        "type": "string",
        "user_input": "poly4"
      },
      "structure_file_name": {
        "default": "your_structure_file_name",
        "title": "Structure File Name",
        "type": "string",
        "agent_input": "7_0.vasp",
        "user_input": "7_0.vasp"
      },
      "work_path": {
        "default": ".",
        "title": "Work Path",
        "type": "string",
        "agent_input": "C:\\Users\\lonya\\PycharmProjects\\dptb-pilot\\workspace\\jPAKkXkMuXfC5oyf3hwPKOW1N4xHa1WV\\files",
        "user_input": "C:\\Users\\lonya\\PycharmProjects\\dptb-pilot\\workspace\\jPAKkXkMuXfC5oyf3hwPKOW1N4xHa1WV\\files"
      },
      "executor": {
        "anyOf": [
          {
            "additionalProperties": true,
            "type": "object"
          },
          {
            "type": "null"
          }
        ],
        "default": null,
        "title": "Executor"
      },
      "storage": {
        "anyOf": [
          {
            "additionalProperties": true,
            "type": "object"
          },
          {
            "type": "null"
          }
        ],
        "default": null,
        "title": "Storage"
      }
    }
  },
  "parameters": {}
}
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] ================================================================================
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] ================================================================================
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [AutoGenerate] 开始自动生成 Executor 和 Storage 参数
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [AutoGenerate] 机器类型: Slurm
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [AutoGenerate] 机器配置: {
  "remote_root": "/public4/home/a0s000985/yyt/CNT/press_band_dptb",
  "hostname": "ssh.cn-zhongwei-1.paracloud.com",
  "username": "a0s000985@BSCC-A",
  "key_filename": "/root/.ssh/id_ed25519",
  "number_node": "1",
  "gpu_per_node": "0",
  "cpu_per_node": "64",
  "queue_name": "amd_256",
  "custom_flags": "source /public4/home/a0s000985/yyt/envs/negf.sh"
}
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [AutoGenerate] Slurm Executor 配置已生成
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [AutoGenerate] Executor: {
  "type": "dispatcher",
  "machine": {
    "batch_type": "Slurm",
    "context_type": "SSHContext",
    "local_root": "./",
    "remote_root": "/public4/home/a0s000985/yyt/CNT/press_band_dptb",
    "remote_profile": {
      "hostname": "ssh.cn-zhongwei-1.paracloud.com",
      "username": "a0s000985@BSCC-A",
      "timeout": 600,
      "port": 22,
      "key_filename": "/root/.ssh/id_ed25519"
    }
  },
  "resource": {
    "number_node": 1,
    "gpu_per_node": 0,
    "cpu_per_node": 64,
    "queue_name": "amd_256",
    "custom_flags": [
      "source /public4/home/a0s000985/yyt/envs/negf.sh",
      ""
    ],
    "source_list": [],
    "module_list": []
  }
}
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [AutoGenerate] 完成 Executor 和 Storage 参数自动生成
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] ================================================================================
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] ================================================================================
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [ModifyParams] 生成 Executor 和 Storage 后的Schema: {
  "name": "band_with_baseline_model",
  "description": "\n    使用基准模型预测结构能带。\n\n    参数:\n        basemodel: 使用的baseline model类型。可选：poly4，poly2\n        structure_file_name: 输入的结构文件路径。结 
构文件应为vasp的格式\n        work_path: 能带信息的保存路径。注意应该是文件夹而不是文件。\n\n    返回:\n        包含能带文件路径的字典。\n\n    抛出:\n        AssumptionError: 某些数输入不合规。\n        RuntimeError: 写入配置文件失败。\n    ",
  "input_schema": {
    "properties": {
      "basemodel": {
        "default": "poly4",
        "title": "Basemodel",
        "type": "string",
        "user_input": "poly4"
      },
      "structure_file_name": {
        "default": "your_structure_file_name",
        "title": "Structure File Name",
        "type": "string",
        "agent_input": "7_0.vasp",
        "user_input": "7_0.vasp"
      },
      "work_path": {
        "default": ".",
        "title": "Work Path",
        "type": "string",
        "agent_input": "C:\\Users\\lonya\\PycharmProjects\\dptb-pilot\\workspace\\jPAKkXkMuXfC5oyf3hwPKOW1N4xHa1WV\\files",
        "user_input": "C:\\Users\\lonya\\PycharmProjects\\dptb-pilot\\workspace\\jPAKkXkMuXfC5oyf3hwPKOW1N4xHa1WV\\files"
      },
      "executor": {
        "anyOf": [
          {
            "additionalProperties": true,
            "type": "object"
          },
          {
            "type": "null"
          }
        ],
        "default": null,
        "title": "Executor",
        "user_input": {
          "type": "dispatcher",
          "machine": {
            "batch_type": "Slurm",
            "context_type": "SSHContext",
            "local_root": "./",
            "remote_root": "/public4/home/a0s000985/yyt/CNT/press_band_dptb",
            "remote_profile": {
              "hostname": "ssh.cn-zhongwei-1.paracloud.com",
              "username": "a0s000985@BSCC-A",
              "timeout": 600,
              "port": 22,
              "key_filename": "/root/.ssh/id_ed25519"
            }
          },
          "resource": {
            "number_node": 1,
            "gpu_per_node": 0,
            "cpu_per_node": 64,
            "queue_name": "amd_256",
            "custom_flags": [
              "source /public4/home/a0s000985/yyt/envs/negf.sh",
              ""
            ],
            "source_list": [],
            "module_list": []
          }
        }
      },
      "storage": {
        "anyOf": [
          {
            "additionalProperties": true,
            "type": "object"
          },
          {
            "type": "null"
          }
        ],
        "default": null,
        "title": "Storage"
      }
    }
  },
  "parameters": {}
}
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] ================================================================================
[2026-01-22 20:17:35] [INFO] [dptb_pilot.core.guardrail] ================================================================================
[2026-01-22 20:17:35] [INFO] [dptb_pilot.core.guardrail] [ExtractArguments] 开始提取参数
[2026-01-22 20:17:35] [INFO] [dptb_pilot.core.guardrail] [ExtractArguments] 工具名称: band_with_baseline_model
[2026-01-22 20:17:35] [INFO] [dptb_pilot.core.guardrail] [ExtractArguments] 最终参数: {'basemodel': 'poly4', 'structure_file_name': '7_0.vasp', 'work_path': 'C:\\Users\\lonya\\Pychar
mProjects\\dptb-pilot\\workspace\\jPAKkXkMuXfC5oyf3hwPKOW1N4xHa1WV\\files', 'executor': {'type': 'dispatcher', 'machine': {'batch_type': 'Slurm', 'context_type': 'SSHContext', 'local
_root': './', 'remote_root': '/public4/home/a0s000985/yyt/CNT/press_band_dptb', 'remote_profile': {'hostname': 'ssh.cn-zhongwei-1.paracloud.com', 'username': 'a0s000985@BSCC-A', 'tim
eout': 600, 'port': 22, 'key_filename': '/root/.ssh/id_ed25519'}}, 'resource': {'number_node': 1, 'gpu_per_node': 0, 'cpu_per_node': 64, 'queue_name': 'amd_256', 'custom_flags': ['source /public4/home/a0s000985/yyt/envs/negf.sh', ''], 'source_list': [], 'module_list': []}}}
[2026-01-22 20:17:35] [INFO] [dptb_pilot.core.guardrail] ================================================================================
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [ModifyParams] 提取后的参数: {'basemodel': 'poly4', 'structure_file_name': '7_0.vasp', 'work_path': 'C:\\Users\\lonya\\PycharmPro
jects\\dptb-pilot\\workspace\\jPAKkXkMuXfC5oyf3hwPKOW1N4xHa1WV\\files', 'executor': {'type': 'dispatcher', 'machine': {'batch_type': 'Slurm', 'context_type': 'SSHContext', 'local_roo
t': './', 'remote_root': '/public4/home/a0s000985/yyt/CNT/press_band_dptb', 'remote_profile': {'hostname': 'ssh.cn-zhongwei-1.paracloud.com', 'username': 'a0s000985@BSCC-A', 'timeout
': 600, 'port': 22, 'key_filename': '/root/.ssh/id_ed25519'}}, 'resource': {'number_node': 1, 'gpu_per_node': 0, 'cpu_per_node': 64, 'queue_name': 'amd_256', 'custom_flags': ['source /public4/home/a0s000985/yyt/envs/negf.sh', ''], 'source_list': [], 'module_list': []}}}
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [ModifyParams] Executor 参数: None
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [ModifyParams] Storage 参数: None
[2026-01-22 20:17:35] [INFO] [dptb_pilot.server.app] [ModifyParams] 触发事件，恢复 agent 执行
```
+ 现在我需要你实现单元测试，实现一个简单的单元测试，并且实现一个用前端调用生产Executor和Storage并正确传参的单元测试，确定单元测试可以运行并能正确传参。