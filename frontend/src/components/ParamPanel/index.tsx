import { useState, useEffect } from 'react';
import {
  Card,
  Select,
  Radio,
  Button,
  Input,
  Form,
  Space,
  Divider,
  Typography,
  InputNumber,
  Switch,
  message,
  Alert,
  Tooltip
} from 'antd';
import {
  ToolOutlined,
  SettingOutlined,
  EditOutlined,
  CheckOutlined,
  InfoCircleOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';
// import JSONInput from 'react-json-editor-ajrm';
// import locale from 'react-json-editor-ajrm/locale/en';

import { useApp } from '../../contexts/AppContext';
import type { ToolSchema, PropertySchema, ExecutionMode, ModifyMode } from '../../types';

const { Title, Text } = Typography;
// const { Panel } = Collapse;
const { Option } = Select;
const { TextArea } = Input;

function ParamPanel() {
  const { state, actions } = useApp();
  const [form] = Form.useForm();
  const [currentSchema, setCurrentSchema] = useState<ToolSchema | null>(null);
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('Local');
  const [modifyMode, setModifyMode] = useState<ModifyMode>('individual');
  const [jsonText, setJsonText] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [bohrConfig, setBohrConfig] = useState({
    username: '',
    password: '',
    project_id: ''
  });
  const [lastSchemaHash, setLastSchemaHash] = useState<string>('');

  useEffect(() => {
    // 获取当前需要修改的参数schema
    const checkForSchema = async () => {
      if (!state.userId) return;

      try {
        const schema = await actions.getCurrentSchema();
        if (schema && Object.keys(schema).length > 0) {
          // 计算schema的hash值来检测变化
          const schemaHash = JSON.stringify(schema);

          // 只有当schema发生变化时才更新UI
          if (schemaHash !== lastSchemaHash) {
            setCurrentSchema(schema);
            setLastSchemaHash(schemaHash);

            if (modifyMode === 'json') {
              setJsonText(JSON.stringify(schema, null, 2));
            } else {
              // 填充表单数据
              const formData: any = {};
              if (schema.input_schema?.properties) {
                Object.entries(schema.input_schema.properties).forEach(([key, prop]: [string, any]) => {
                  formData[key] = prop.agent_input || prop.default || '';
                });
              }
              form.setFieldsValue(formData);
            }
          }
        } else {
          if (lastSchemaHash !== '') {
            setCurrentSchema(null);
            setLastSchemaHash('');
          }
        }
      } catch (error) {
        console.error('获取参数schema失败:', error);
      }
    };

    const interval = setInterval(checkForSchema, 1000);
    return () => clearInterval(interval);
  }, [state.userId, actions, form, modifyMode, lastSchemaHash]);

  const handleExecutionModeChange = (value: ExecutionMode) => {
    setExecutionMode(value);
    // 可以在这里触发后端配置更新
    console.log('执行模式变更为:', value);
  };

  const handleModifyModeChange = (e: any) => {
    setModifyMode(e.target.value);
  };

  const handleBohrConfigChange = (field: string, value: string) => {
    setBohrConfig(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmitIndividual = async (values: any) => {
    if (!currentSchema || !state.userId) return;

    setSubmitting(true);
    try {
      // 构建修改后的schema
      const modifiedSchema = {
        ...currentSchema,
        input_schema: {
          properties: {}
        }
      };

      if (currentSchema.input_schema?.properties) {
        Object.entries(currentSchema.input_schema.properties).forEach(([key, prop]) => {
          (modifiedSchema.input_schema.properties as any)[key] = {
            ...prop,
            user_input: values[key]
          };
        });
      }

      await actions.modifyParameters(modifiedSchema);
      message.success('参数已提交');
      setCurrentSchema(null);
      setLastSchemaHash(''); // 重置hash以触发重新检查
      form.resetFields();
    } catch (error) {
      message.error('参数提交失败');
      console.error('Submit error:', error);
    } finally {
      setSubmitting(false);
    }
  };

  const handleSubmitJson = async () => {
    if (!state.userId) return;

    try {
      const modifiedSchema = JSON.parse(jsonText);
      setSubmitting(true);
      await actions.modifyParameters(modifiedSchema);
      message.success('参数已提交');
      setCurrentSchema(null);
      setLastSchemaHash(''); // 重置hash以触发重新检查
      setJsonText('');
    } catch (error) {
      if (error instanceof SyntaxError) {
        message.error('JSON格式错误');
      } else {
        message.error('参数提交失败');
        console.error('Submit error:', error);
      }
    } finally {
      setSubmitting(false);
    }
  };

  const renderFormInput = (_key: string, prop: PropertySchema) => {
    const commonProps = {
      style: { width: '100%' }
    };

    switch (prop.type) {
      case 'string':
        return (
          <Input
            {...commonProps}
            defaultValue={prop.agent_input || prop.default || ''}
            placeholder={prop.default}
          />
        );
      case 'number':
      case 'integer':
        return (
          <InputNumber
            {...commonProps}
            defaultValue={prop.agent_input || prop.default || 0}
            precision={prop.type === 'integer' ? 0 : 2}
            style={{ width: '100%' }}
          />
        );
      case 'boolean':
        return (
          <Switch
            defaultChecked={prop.agent_input || prop.default || false}
          />
        );
      case 'array':
      case 'object':
        return (
          <TextArea
            {...commonProps}
            defaultValue={JSON.stringify(prop.agent_input || prop.default || {}, null, 2)}
            rows={3}
            placeholder="请输入JSON格式数据"
          />
        );
      default:
        return (
          <Input
            {...commonProps}
            defaultValue={prop.agent_input || prop.default || ''}
          />
        );
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <Title level={5} style={{ margin: 0 }}>
          <SettingOutlined style={{ marginRight: '8px' }} />
          修改运行参数
        </Title>
        {currentSchema && (
          <Tooltip title="工具正在等待参数确认">
            <ThunderboltOutlined style={{ color: '#fa8c16' }} />
          </Tooltip>
        )}
      </div>

      <Space direction="vertical" style={{ width: '100%', marginBottom: '16px' }}>
        <div>
          <Text strong style={{ display: 'block', marginBottom: '4px' }}>
            工具执行模式
          </Text>
          <Select
            value={executionMode}
            onChange={handleExecutionModeChange}
            style={{ width: '100%' }}
            size="small"
          >
            <Option value="Local">
              <Tooltip title="在agent部署服务器运行">
                在线运行
              </Tooltip>
            </Option>
            <Option value="Bohr">
              <Tooltip title="作为任务提交到Bohrium">
                玻尔模式
              </Tooltip>
            </Option>
          </Select>
        </div>

        {executionMode === 'Bohr' && (
          <Card size="small" title="玻尔配置" styles={{ body: { padding: '12px' } }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Input
                placeholder="Bohrium用户名"
                size="small"
                value={bohrConfig.username}
                onChange={(e) => handleBohrConfigChange('username', e.target.value)}
              />
              <Input.Password
                placeholder="Bohrium密码"
                size="small"
                value={bohrConfig.password}
                onChange={(e) => handleBohrConfigChange('password', e.target.value)}
              />
              <Input
                placeholder="项目ID"
                size="small"
                value={bohrConfig.project_id}
                onChange={(e) => handleBohrConfigChange('project_id', e.target.value)}
              />
            </Space>
          </Card>
        )}
      </Space>

      <Divider />

      <div style={{ marginBottom: '16px' }}>
        <Text strong style={{ display: 'block', marginBottom: '8px' }}>
          参数修改方式
        </Text>
        <Radio.Group value={modifyMode} onChange={handleModifyModeChange} size="small">
          <Radio.Button value="individual">
            <EditOutlined /> 逐个修改
          </Radio.Button>
          <Radio.Button value="json">
            <ToolOutlined /> 使用JSON
          </Radio.Button>
        </Radio.Group>
      </div>

      {currentSchema ? (
        <>
          {state.pendingToolResponse && (
            <Alert
              message="Agent响应"
              description={
                <div style={{ maxHeight: '200px', overflowY: 'auto', fontSize: '12px' }}>
                  <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                    {state.pendingToolResponse}
                  </pre>
                </div>
              }
              type="info"
              showIcon
              style={{ marginBottom: '16px' }}
            />
          )}
          <Card
            title={`工具: ${currentSchema.name}`}
            size="small"
            styles={{ body: { padding: '16px' } }}
            extra={
              <Tooltip title={currentSchema.description}>
                <InfoCircleOutlined />
              </Tooltip>
            }
          >
          {modifyMode === 'individual' ? (
            <Form
              form={form}
              onFinish={handleSubmitIndividual}
              layout="vertical"
              size="small"
            >
              {currentSchema.input_schema?.properties && Object.entries(currentSchema.input_schema.properties).map(([key, prop]) => (
                <Form.Item
                  key={key}
                  label={
                    <Space>
                      <Text>{prop.title || key}</Text>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        ({prop.type})
                      </Text>
                    </Space>
                  }
                  extra={
                    <div>
                      {prop.default !== undefined && (
                        <Text type="secondary" style={{ fontSize: '11px' }}>
                          默认: {JSON.stringify(prop.default)}
                        </Text>
                      )}
                      {prop.agent_input !== undefined && (
                        <div>
                          <Text type="warning" style={{ fontSize: '11px' }}>
                            Agent值: {JSON.stringify(prop.agent_input)}
                          </Text>
                        </div>
                      )}
                    </div>
                  }
                  name={key}
                >
                  {renderFormInput(key, prop)}
                </Form.Item>
              ))}

              <Form.Item style={{ marginTop: '16px', marginBottom: '0' }}>
                <Button
                  type="primary"
                  htmlType="submit"
                  icon={<CheckOutlined />}
                  loading={submitting}
                  block
                >
                  提交修改
                </Button>
              </Form.Item>
            </Form>
          ) : (
            <Space direction="vertical" style={{ width: '100%' }}>
              <TextArea
                value={jsonText}
                onChange={(e) => setJsonText(e.target.value)}
                placeholder="请输入JSON格式的参数"
                rows={8}
                style={{ fontFamily: 'monospace', fontSize: '12px' }}
              />
              <Button
                type="primary"
                icon={<CheckOutlined />}
                onClick={handleSubmitJson}
                loading={submitting}
                block
              >
                提交JSON
              </Button>
            </Space>
          )}
        </Card>
        </>
      ) : (
        <Alert
          message="等待工具调用"
          description="当Agent调用MCP工具时，此处会显示工具参数供您确认或修改。"
          type="info"
          showIcon
          style={{ fontSize: '12px' }}
        />
      )}
    </div>
  );
}

export default ParamPanel;