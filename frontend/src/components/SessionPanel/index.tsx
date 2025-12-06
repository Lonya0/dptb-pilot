import { useState } from 'react';
import {
  Button,
  List,
  Typography,
  message,
  Modal,
  Input,
  Tooltip,
  Badge
} from 'antd';
import {
  PlusOutlined,
  HistoryOutlined,
  DeleteOutlined,
  EditOutlined,
  ClockCircleOutlined,
  MessageOutlined
} from '@ant-design/icons';

import { useApp } from '../../contexts/AppContext';
import type { ChatSession } from '../../types';

const { Title, Text } = Typography;

function SessionPanel() {
  const { state, actions, dispatch } = useApp();
  const [editingSession, setEditingSession] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
  // const [loading, setLoading] = useState(false);

  // 创建新聊天会话
  const createNewChatSession = async () => {
    try {
      await actions.createNewChatSession();
      message.success('新聊天会话已创建');
    } catch (error) {
      message.error('创建聊天会话失败');
    }
  };

  const switchToChatSession = async (chatId: string) => {
    if (chatId === state.currentChatSession?.chat_id) {
      message.info('当前已是此聊天会话');
      return;
    }

    try {
      await actions.switchToChatSession(chatId);
      message.success('已切换到选中的聊天会话');
    } catch (error) {
      message.error('切换聊天会话失败');
    }
  };

  const editSessionTitle = (chatId: string, currentTitle: string) => {
    setEditingSession(chatId);
    setEditingTitle(currentTitle);
  };

  const saveSessionTitle = (chatId: string) => {
    if (!editingTitle.trim()) {
      message.error('聊天会话标题不能为空');
      return;
    }

    const updatedSessions = state.chatSessions.map(session =>
      session.chat_id === chatId
        ? { ...session, title: editingTitle.trim() }
        : session
    );

    // 更新状态并保存到localStorage
    if (state.userId) {
      try {
        localStorage.setItem(`chat_sessions_${state.userId}`, JSON.stringify(updatedSessions));
      } catch (error) {
        console.error('保存聊天会话失败:', error);
      }
    }
    setEditingSession(null);
    setEditingTitle('');
    message.success('聊天会话标题已更新');
  };

  // const cancelEditTitle = () => {
  //   setEditingSession(null);
  //   setEditingTitle('');
  // };

  const deleteChatSession = (chatId: string) => {
    Modal.confirm({
      title: '删除聊天会话',
      content: '确定要删除这个聊天会话吗？删除后无法恢复。',
      okText: '删除',
      okType: 'danger',
      onOk: () => {
        const updatedSessions = state.chatSessions.filter(session => session.chat_id !== chatId);

        // 保存到localStorage
        if (state.userId) {
          try {
            localStorage.setItem(`chat_sessions_${state.userId}`, JSON.stringify(updatedSessions));
          } catch (error) {
            console.error('保存聊天会话失败:', error);
          }
        }

        // 通过dispatch更新状态
        // const { dispatch } = require('../../contexts/AppContext');
        dispatch({ type: 'SET_CHAT_SESSIONS', payload: updatedSessions });

        message.success('聊天会话已删除');

        // 如果删除的是当前聊天会话，创建一个新的
        if (chatId === state.currentChatSession?.chat_id) {
          actions.createNewChatSession();
        }
      }
    });
  };

  const formatLastActive = (lastActive: string) => {
    try {
      const date = new Date(lastActive);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMs / 3600000);
      const diffDays = Math.floor(diffMs / 86400000);

      if (diffMins < 1) return '刚刚';
      if (diffMins < 60) return `${diffMins}分钟前`;
      if (diffHours < 24) return `${diffHours}小时前`;
      if (diffDays < 7) return `${diffDays}天前`;

      return date.toLocaleDateString();
    } catch {
      return '未知时间';
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <Title level={5} style={{ margin: 0 }}>
          <HistoryOutlined style={{ marginRight: '8px' }} />
          会话记录
        </Title>
        <Tooltip title="创建新会话">
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={createNewChatSession}
            size="small"
          />
        </Tooltip>
      </div>

      <List
        size="small"
        dataSource={state.chatSessions}
        locale={{ emptyText: '暂无聊天会话记录' }}
        renderItem={(session: ChatSession) => (
          <List.Item
            style={{
              padding: '8px 0',
              borderBottom: '1px solid #f0f0f0',
              cursor: 'pointer',
              backgroundColor: session.chat_id === state.currentChatSession?.chat_id ? '#e6f7ff' : 'transparent'
            }}
            onClick={() => switchToChatSession(session.chat_id)}
            actions={[
              <Tooltip title="编辑标题" key="edit">
                <Button
                  type="text"
                  icon={<EditOutlined />}
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    editSessionTitle(session.chat_id, session.title);
                  }}
                />
              </Tooltip>,
              <Tooltip title="删除聊天会话" key="delete">
                <Button
                  type="text"
                  danger
                  icon={<DeleteOutlined />}
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteChatSession(session.chat_id);
                  }}
                />
              </Tooltip>
            ]}
          >
            <List.Item.Meta
              avatar={
                session.chat_id === state.currentChatSession?.chat_id ? (
                  <Badge dot color="#52c41a">
                    <MessageOutlined style={{ color: '#1677ff' }} />
                  </Badge>
                ) : (
                  <MessageOutlined style={{ color: '#8c8c8c' }} />
                )
              }
              title={
                editingSession === session.chat_id ? (
                  <Input
                    value={editingTitle}
                    onChange={(e) => setEditingTitle(e.target.value)}
                    onPressEnter={() => saveSessionTitle(session.chat_id)}
                    onBlur={() => saveSessionTitle(session.chat_id)}
                    size="small"
                    style={{ width: '120px' }}
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <Text strong style={{ fontSize: '14px' }}>
                    {session.title}
                  </Text>
                )
              }
              description={
                <div onClick={(e) => e.stopPropagation()}>
                  <div style={{
                    fontSize: '12px',
                    color: '#666',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    maxWidth: '150px'
                  }}>
                    {session.history.length > 0 ? session.history[session.history.length - 1].content.substring(0, 50) + (session.history[session.history.length - 1].content.length > 50 ? '...' : '') : '新对话'}
                  </div>
                  <div style={{
                    fontSize: '11px',
                    color: '#999',
                    marginTop: '2px',
                    display: 'flex',
                    alignItems: 'center'
                  }}>
                    <ClockCircleOutlined style={{ marginRight: '4px' }} />
                    {formatLastActive(session.last_active)}
                    <span style={{ marginLeft: '8px' }}>
                      {session.message_count} 条消息
                    </span>
                  </div>
                </div>
              }
            />
          </List.Item>
        )}
      />

      {state.chatSessions.length > 0 && (
        <div style={{ marginTop: '12px', textAlign: 'center' }}>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            共 {state.chatSessions.length} 个聊天会话
          </Text>
        </div>
      )}
    </div>
  );
}

export default SessionPanel;