import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Input, Button, Typography, Form, Alert, Spin, Space, Divider } from 'antd';
import { UserOutlined, KeyOutlined, ReloadOutlined, RobotOutlined } from '@ant-design/icons';
import { useApp } from '../../contexts/AppContext';

const { Title, Paragraph, Text } = Typography;

function generateRandomString(length = 32): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

function Login() {
  const navigate = useNavigate();
  const { state, actions, dispatch } = useApp();
  const [sessionId, setSessionId] = useState('');
  const [form] = Form.useForm();

  useEffect(() => {
    // 如果已经认证，重定向到聊天页面
    if (state.isAuthenticated) {
      navigate('/chat');
    }
  }, [state.isAuthenticated, navigate]);

  useEffect(() => {
    // 初始化为空，不自动生成ID
    setSessionId('');
    form.setFieldsValue({ session_id: '' });
  }, [form]);

  const handleGenerateRandom = () => {
    const newId = generateRandomString(32);
    setSessionId(newId);
    form.setFieldsValue({ session_id: newId });
  };

  const handleSubmit = async (values: { session_id: string }) => {
    if (!values.session_id || values.session_id.length !== 32) {
      return;
    }

    try {
      await actions.login(values.session_id);
      navigate('/chat');
    } catch (error) {
      // 错误已通过context处理，这里不需要额外处理
      console.error('Login submit error:', error);
    }
  };

  const handleSessionIdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    // 只允许字母数字字符，最大32位
    const filteredValue = value.replace(/[^a-zA-Z0-9]/g, '').slice(0, 32);
    setSessionId(filteredValue);
    form.setFieldsValue({ session_id: filteredValue });
  };

  if (state.loading && !state.error) {
    return (
      <div style={{
        height: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center'
      }}>
        <Spin size="large" tip="正在连接服务器..." />
      </div>
    );
  }

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px'
    }}>
      <Card
        style={{
          width: '100%',
          maxWidth: 500,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          borderRadius: '12px'
        }}
        bodyStyle={{ padding: '32px' }}
      >
        <div style={{ textAlign: 'center', marginBottom: '24px' }}>
          <RobotOutlined style={{ fontSize: '48px', color: '#1677ff', marginBottom: '16px' }} />
          <Title level={2} style={{ margin: '0', color: '#262626' }}>
            Better AIM
          </Title>
          <Paragraph style={{ color: '#8c8c8c', margin: '8px 0' }}>
            AI Agent 交互平台
          </Paragraph>
        </div>

        <Divider>创建会话</Divider>

        {state.error && (
          <Alert
            message="连接错误"
            description={state.error}
            type="error"
            showIcon
            closable
            onClose={() => {
              // 清空错误状态，不触发登录
              dispatch({ type: 'SET_ERROR', payload: null });
            }}
            style={{ marginBottom: '16px' }}
          />
        )}

        <Form
          form={form}
          onFinish={handleSubmit}
          layout="vertical"
          size="large"
        >
          <Form.Item
            label="会话ID"
            name="session_id"
            rules={[
              { required: true, message: '请输入会话ID' },
              { len: 32, message: '会话ID需要为长度为32的任意字符' }
            ]}
            extra="输入或自动生成32位任意字符串作为您的专属会话ID，使用相同ID可以访问此前的历史记录，历史记录在一小时后会被自动清除，请不要传播您专属的ID！"
          >
            <Space.Compact style={{ width: '100%' }}>
              <Input
                placeholder="请输入32位任意字符串"
                prefix={<KeyOutlined />}
                value={sessionId}
                onChange={handleSessionIdChange}
                style={{ flex: 1 }}
              />
              <Button
                type="primary"
                icon={<ReloadOutlined />}
                onClick={handleGenerateRandom}
                title="随机生成"
              >
                随机生成
              </Button>
            </Space.Compact>
          </Form.Item>

          <Form.Item style={{ marginBottom: '0' }}>
            <Button
              type="primary"
              htmlType="submit"
              icon={<UserOutlined />}
              loading={state.loading}
              block
              style={{ height: '48px', fontSize: '16px' }}
            >
              进入会话
            </Button>
          </Form.Item>
        </Form>

        {state.config && (
          <div style={{ marginTop: '24px', padding: '16px', background: '#f5f5f5', borderRadius: '8px' }}>
            <Text strong>当前服务信息：</Text>
            <div style={{ marginTop: '8px' }}>
              <div style={{ fontSize: '14px', color: '#666' }}>
                <div>Agent: {state.config.agent_info.name}</div>
                <div>描述: {state.config.agent_info.description}</div>
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

export default Login;