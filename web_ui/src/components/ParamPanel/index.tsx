import { useState, useEffect } from 'react';
import {
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
  CheckOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
// import JSONInput from 'react-json-editor-ajrm';
// import locale from 'react-json-editor-ajrm/locale/en';

import { useApp } from '../../contexts/AppContext';
import type { ToolSchema, PropertySchema, ExecutionMode, ModifyMode } from '../../types';
import { translations } from '../../utils/i18n';

const { Text } = Typography;
// const { Panel } = Collapse;
const { Option } = Select;
const { TextArea } = Input;

function ParamPanel() {
  const { state, actions } = useApp();
  const t = translations[state.language];
  const [form] = Form.useForm();
  const [currentSchema, setCurrentSchema] = useState<ToolSchema | null>(null);
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('Local');
  const [modifyMode, setModifyMode] = useState<ModifyMode>('individual');
  const [jsonText, setJsonText] = useState('');
  const [submitting, setSubmitting] = useState(false);

  // 从localStorage加载bohrium配置
  const [bohrConfig, setBohrConfig] = useState(() => {
    const saved = localStorage.getItem('bohrConfig');
    return saved ? JSON.parse(saved) : {
      username: '',
      password: '',
      project_id: ''
    };
  });
  const [lastSchemaHash, setLastSchemaHash] = useState<string>('');

  // 持久化保存bohrium配置到localStorage
  useEffect(() => {
    localStorage.setItem('bohrConfig', JSON.stringify(bohrConfig));
  }, [bohrConfig]);

  // 从localStorage加载executionMode
  useEffect(() => {
    const savedMode = localStorage.getItem('executionMode') as ExecutionMode;
    if (savedMode && (savedMode === 'Local' || savedMode === 'Bohr')) {
      setExecutionMode(savedMode);
    }
  }, []);

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
    // 保存到localStorage
    localStorage.setItem('executionMode', value);
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
          // 如果是Bohr模式且参数名为Executor或Storage，注入配置
          if (executionMode === 'Bohr') {
            if (key === 'Executor') {
              (modifiedSchema.input_schema.properties as any)[key] = {
                ...prop,
                user_input: {
                  type: 'dispatcher',
                  machine: {
                    batch_type: 'Bohrium',
                    context_type: 'Bohrium',
                    remote_profile: {
                      email: bohrConfig.username,
                      password: bohrConfig.password,
                      program_id: parseInt(bohrConfig.project_id),
                      input_data: {
                        image_name: 'registry.dp.tech/dptech/dp/native/prod-19853/dpa-mcp:0.0.0',
                        job_type: 'container',
                        platform: 'ali',
                        scass_type: '1 * NVIDIA V100_32g'
                      }
                    }
                  }
                }
              };
            } else if (key === 'Storage') {
              (modifiedSchema.input_schema.properties as any)[key] = {
                ...prop,
                user_input: {
                  type: 'bohrium',
                  username: bohrConfig.username,
                  password: bohrConfig.password,
                  program_id: parseInt(bohrConfig.project_id)
                }
              };
            } else {
              // 其他参数使用用户输入的值
              (modifiedSchema.input_schema.properties as any)[key] = {
                ...prop,
                user_input: values[key]
              };
            }
          } else {
            // Local模式，使用用户输入的值
            (modifiedSchema.input_schema.properties as any)[key] = {
              ...prop,
              user_input: values[key]
            };
          }
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
      {/* Execution Mode Section */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <Text style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            {t.executionMode}
          </Text>
        </div>
        <Select
          value={executionMode}
          onChange={handleExecutionModeChange}
          style={{ width: '100%' }}
          className="glass-select-dense"
          dropdownStyle={{
            backgroundColor: '#1e293b',
            border: '1px solid rgba(255,255,255,0.1)'
          }}
        >
          <Option value="Local" style={{ color: '#e2e8f0' }}>{t.localExecution}</Option>
          <Option value="Bohr" style={{ color: '#e2e8f0' }}>{t.bohriumCloud}</Option>
        </Select>
      </div>

      {/* Bohrium Config (Conditional) */}
      {executionMode === 'Bohr' && (
        <div className="glass-card" style={{ padding: '12px', marginBottom: '24px', background: 'rgba(30, 41, 59, 0.5)' }}>
          <Text style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '8px' }}>
            {t.bohrConfig}
          </Text>
          <Space direction="vertical" style={{ width: '100%', gap: '8px' }}>
            <Input
              placeholder="Username"
              size="small"
              value={bohrConfig.username}
              onChange={(e) => handleBohrConfigChange('username', e.target.value)}
              className="glass-input-dense"
            />
            <Input.Password
              placeholder="Password"
              size="small"
              value={bohrConfig.password}
              onChange={(e) => handleBohrConfigChange('password', e.target.value)}
              className="glass-input-dense"
              style={{
                backgroundColor: 'rgba(15, 23, 42, 0.6) !important',
                color: '#f1f5f9 !important'
              }}
              styles={{
                input: {
                  backgroundColor: 'transparent',
                  color: '#f1f5f9'
                }
              }}
            />
            <Input
              placeholder="Project ID"
              size="small"
              value={bohrConfig.project_id}
              onChange={(e) => handleBohrConfigChange('project_id', e.target.value)}
              className="glass-input-dense"
            />
          </Space>
        </div>
      )}

      {/* Agent Status Bar */}
      <div style={{
        backgroundColor: 'rgba(30, 41, 59, 0.5)', // bg-slate-800/50
        border: '1px solid rgba(51, 65, 85, 0.5)', // border-slate-700/50
        borderRadius: '8px',
        padding: '10px 12px',
        marginBottom: '24px',
        display: 'flex',
        alignItems: 'center',
        gap: '10px'
      }}>
        <div style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor: (() => {
            if (state.responding) return '#eab308'; // yellow-500 - 工作中
            if (currentSchema && Object.keys(currentSchema).length > 0) return '#f97316'; // orange-500 - 等待用户传参
            return '#22c55e'; // green-500 - 空闲
          })(),
          boxShadow: (() => {
            if (state.responding) return '0 0 8px rgba(234, 179, 8, 0.4)';
            if (currentSchema && Object.keys(currentSchema).length > 0) return '0 0 8px rgba(249, 115, 22, 0.4)';
            return '0 0 8px rgba(34, 197, 94, 0.4)';
          })()
        }} />
        <Text style={{ color: '#94a3b8', fontSize: '12px', fontWeight: 500 }}>
          {t.agentStatus}: <span style={{ color: '#cbd5e1' }}>
            {state.responding
              ? t.statusWorking
              : (currentSchema && Object.keys(currentSchema).length > 0)
                ? '等待用户传参'
                : t.statusIdle
            }
          </span>
        </Text>
      </div>

      <Divider style={{ borderColor: 'rgba(255,255,255,0.05)', margin: '24px 0' }} />

      {/* Parameter Modification Header & Toggle */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <Text style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          {t.paramModifyMode}
        </Text>
        <Radio.Group 
          value={modifyMode} 
          onChange={handleModifyModeChange} 
          size="small" 
          buttonStyle="solid"
          className="dense-toggle"
        >
          <Radio.Button value="individual" style={{ fontSize: '11px', padding: '0 8px' }}>{t.individual}</Radio.Button>
          <Radio.Button value="json" style={{ fontSize: '11px', padding: '0 8px' }}>{t.json}</Radio.Button>
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
          <div className="glass-card" style={{ padding: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <Text strong>工具: {currentSchema.name}</Text>
              <Tooltip title={currentSchema.description}>
                <InfoCircleOutlined />
              </Tooltip>
            </div>
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
                  {t.submit}
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
                {t.submitJson}
              </Button>
            </Space>
          )}
          </div>
        </>
      ) : (
        <Alert
          message={t.waitingForTool}
          description={t.waitingForToolDesc}
          type="info"
          showIcon
          style={{ fontSize: '12px' }}
        />
      )}
    </div>
  );
}

export default ParamPanel;