import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Layout,
  Button,
  Input,
  Typography,
  Space,
  message,
  Spin,
  Alert,
  Tooltip,
  Upload,
  Avatar,
  Tag
} from 'antd';
import {
  SendOutlined,
  ClearOutlined,
  LogoutOutlined,
  UserOutlined,
  RobotOutlined,
  PaperClipOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  SettingOutlined,
  SearchOutlined,
  GlobalOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  DoubleRightOutlined,
  DoubleLeftOutlined
} from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import StructureViewer from '../StructureViewer';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

import { useApp } from '../../contexts/AppContext';
import FilePanel from '../FilePanel';
import ParamPanel from '../ParamPanel';
import SessionPanel from '../SessionPanel';
import { translations } from '../../utils/i18n';

const { Header, Content, Sider } = Layout;
const { Title, Text } = Typography;
const { TextArea } = Input;

function Chat() {
  const navigate = useNavigate();
  const { state, actions } = useApp();
  const [inputValue, setInputValue] = useState('');
  const [activeTab, setActiveTab] = useState('params');

  // 侧边栏状态管理
  const [leftSidebarCollapsed, setLeftSidebarCollapsed] = useState(false);
  const [rightSidebarCollapsed, setRightSidebarCollapsed] = useState(false);

  // 移动端检测
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const t = translations[state.language];

  // Shortcut Cards Data
  const SHORTCUT_CARDS = [
    {
      icon: <ExperimentOutlined style={{ fontSize: '24px', color: '#0ea5e9' }} />,
      title: t.analyzeStructures,
      desc: t.analyzeStructuresDesc,
      prompt: '请帮我分析一下上传的晶体结构文件'
    },
    {
      icon: <LineChartOutlined style={{ fontSize: '24px', color: '#8b5cf6' }} />,
      title: t.calculateBands,
      desc: t.calculateBandsDesc,
      prompt: '请帮我计算并绘制能带图'
    },
    {
      icon: <SettingOutlined style={{ fontSize: '24px', color: '#10b981' }} />,
      title: t.generateConfigs,
      desc: t.generateConfigsDesc,
      prompt: '请帮我生成DeepTB训练配置文件'
    },
    {
      icon: <SearchOutlined style={{ fontSize: '24px', color: '#f59e0b' }} />,
      title: t.searchMaterials,
      desc: t.searchMaterialsDesc,
      prompt: '请帮我搜索相关的材料数据'
    }
  ];

  useEffect(() => {
    if (!state.isAuthenticated) {
      navigate('/login');
      return;
    }
  }, [state.isAuthenticated, navigate]);

  useEffect(() => {
    scrollToBottom();
  }, [state.currentChatSession?.history]);

  // 窗口大小变化监听
  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      const newIsMobile = width < 768;

      setIsMobile(newIsMobile);

      // 移动端自动收起侧边栏
      if (newIsMobile) {
        setLeftSidebarCollapsed(true);
        setRightSidebarCollapsed(true);
      } else {
        setLeftSidebarCollapsed(false);
        setRightSidebarCollapsed(false);
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize(); // 初始化调用

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // 检测函数调用并自动展开右侧栏
  useEffect(() => {
    if (state.currentChatSession?.history && isMobile) {
      const lastMessage = state.currentChatSession.history[state.currentChatSession.history.length - 1];
      if (lastMessage?.content?.includes('function_call') || lastMessage?.content?.includes('tool_calls')) {
        setRightSidebarCollapsed(false);
      }
    }
  }, [state.currentChatSession?.history, isMobile]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (text?: string) => {
    const content = text || inputValue;
    if (!content.trim() || !state.userId || !state.currentChatSession) return;

    setInputValue('');
    try {
      await actions.sendMessage(content);
    } catch (error) {
      message.error(t.sendFailed);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleClearChat = async () => {
    try {
      await actions.clearCurrentChatHistory();
      message.success(t.clearSuccess);
    } catch (error) {
      message.error(t.clearFailed);
    }
  };

  const handleLogout = () => {
    actions.logout();
    navigate('/login');
  };

  const [uploading, setUploading] = useState(false);

  const handleUpload = async (file: File) => {
    if (!state.userId) {
      message.error(t.loginRequired);
      return false;
    }

    if (file.size > 10 * 1024 * 1024) {
      message.error(t.fileSizeLimit);
      return false;
    }

    setUploading(true);
    try {
      await actions.uploadFiles([file]);
      message.success(`${t.uploadSuccess}: ${file.name}`);
    } catch (error) {
      message.error(`${t.uploadFailed}: ${file.name}`);
    } finally {
      setUploading(false);
    }

    return false;
  };

  const formatMessage = (content: string) => {
    const visualizeRegex = /:::visualize\n([\s\S]*?)\n:::/;
    const match = content.match(visualizeRegex);

    if (match) {
      try {
        const jsonStr = match[1];
        const { format, data } = JSON.parse(jsonStr);
        const parts = content.split(match[0]);

        return (
          <div>
            {parts[0] && <ReactMarkdown components={{
              code(props) {
                const {children, className, node, ...rest} = props;
                const match = /language-(\w+)/.exec(className || '');
                const isInline = (props as any).inline;
                const { ref, ...propsWithoutRef } = rest as any;
                return !isInline && match ? (
                  <SyntaxHighlighter
                    {...propsWithoutRef}
                    children={String(children).replace(/\n$/, '')}
                    style={vscDarkPlus}
                    language={match[1]}
                    PreTag="div"
                  />
                ) : (
                  <code {...rest} className={className}>
                    {children}
                  </code>
                )
              },
              img(props) {
                return <img {...props} style={{ maxWidth: '100%', borderRadius: '8px' }} />;
              }
            }}>{parts[0]}</ReactMarkdown>}
            
            <div style={{ margin: '10px 0' }}>
              <StructureViewer data={data} format={format} />
            </div>
            
            {parts[1] && <ReactMarkdown components={{
              code(props) {
                const {children, className, node, ...rest} = props;
                const match = /language-(\w+)/.exec(className || '');
                const isInline = (props as any).inline;
                const { ref, ...propsWithoutRef } = rest as any;
                return !isInline && match ? (
                  <SyntaxHighlighter
                    {...propsWithoutRef}
                    children={String(children).replace(/\n$/, '')}
                    style={vscDarkPlus}
                    language={match[1]}
                    PreTag="div"
                  />
                ) : (
                  <code {...rest} className={className}>
                    {children}
                  </code>
                )
              },
              img(props) {
                return <img {...props} style={{ maxWidth: '100%', borderRadius: '8px' }} />;
              }
            }}>{parts[1]}</ReactMarkdown>}
          </div>
        );
      } catch (e) {
        console.error("Failed to parse visualization block", e);
      }
    }

    return (
      <ReactMarkdown
        components={{
          code(props) {
            const {children, className, node, ...rest} = props;
            const match = /language-(\w+)/.exec(className || '');
            const isInline = (props as any).inline;
            const { ref, ...propsWithoutRef } = rest as any;
            return !isInline && match ? (
              <SyntaxHighlighter
                {...propsWithoutRef}
                children={String(children).replace(/\n$/, '')}
                style={vscDarkPlus}
                language={match[1]}
                PreTag="div"
              />
            ) : (
              <code {...rest} className={className}>
                {children}
              </code>
            )
          },
          img(props) {
            return <img {...props} style={{ maxWidth: '100%', borderRadius: '8px' }} />;
          }
        }}
      >
        {content}
      </ReactMarkdown>
    );
  };

  const renderMessage = (msg: any, index: number) => {
    const isUser = msg.role === 'user';
    const isComplete = !!msg.timestamp;

    return (
      <div
        key={index}
        style={{
          display: 'flex',
          justifyContent: isUser ? 'flex-end' : 'flex-start',
          marginBottom: '24px'
        }}
      >
        <div style={{
          display: 'flex',
          alignItems: 'flex-start',
          maxWidth: '75%',
          width: 'auto'
        }}>
          {!isUser && (
            <Avatar 
              icon={<RobotOutlined />}
              style={{ marginRight: '16px', marginTop: '4px', backgroundColor: '#1e293b', flexShrink: 0 }} 
            />
          )}
          <div className={`chat-message ${isUser ? 'user' : 'assistant'}`} style={{
            backgroundColor: isUser ? '#2563eb' : '#1e293b', // blue-600 : slate-800
            color: isUser ? '#ffffff' : '#e2e8f0', // white : slate-200
            padding: '12px 16px',
            borderRadius: '12px',
            boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)'
          }}>
            {isUser ? (
              <Text style={{ color: 'white' }}>{msg.content}</Text>
            ) : (
              <div className="markdown-content">
                {formatMessage(msg.content)}
              </div>
            )}
            {!isComplete && state.responding && (
              <div style={{ display: 'inline-block', marginLeft: '8px' }}>
                <Spin size="small" />
              </div>
            )}
          </div>
          {isUser && (
            <Avatar 
              icon={<UserOutlined />} 
              style={{ marginLeft: '16px', marginTop: '4px', backgroundColor: '#0ea5e9', flexShrink: 0 }} 
            />
          )}
        </div>
      </div>
    );
  };

  if (!state.isAuthenticated) {
    return (
      <div style={{
        height: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        background: '#020617' // slate-950
      }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <Layout style={{ height: '100vh', background: '#020617' }}> {/* slate-950 */}
      {/* Top Navigation */}
      <Header style={{
        background: 'rgba(15, 23, 42, 0.8)', // slate-900/80
        backdropFilter: 'blur(16px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
        padding: '0 24px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        height: '64px',
        zIndex: 10
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <img 
            src="/pilot_logo_white.png" 
            alt="DeepTB Pilot" 
            style={{ 
              height: '32px', 
              filter: 'drop-shadow(0 0 5px rgba(59, 130, 246, 0.5))' 
            }} 
          />
        </div>

        <Space size="small">
          {/* 桌面端显示所有按钮 */}
          {!isMobile && (
            <>
              <Button
                icon={<GlobalOutlined />}
                onClick={actions.toggleLanguage}
                className="glass-btn"
                shape="round"
                size="small"
                style={{ color: '#94a3b8', borderColor: 'rgba(255,255,255,0.1)' }}
              >
                {state.language === 'zh' ? '中 / En' : 'En / 中'}
              </Button>
              <Tooltip title={`User ID: ${state.userId}`}>
                <Tag icon={<UserOutlined />} color="blue" style={{
                  margin: 0,
                  padding: '6px 12px',
                  borderRadius: '20px',
                  background: 'rgba(14, 165, 233, 0.1)',
                  border: '1px solid rgba(14, 165, 233, 0.2)',
                  color: '#0ea5e9'
                }}>
                  {state.userId?.slice(0, 4)}...{state.userId?.slice(-4)}
                </Tag>
              </Tooltip>
              <Tooltip title={t.clearChat}>
                <Button
                  icon={<ClearOutlined />}
                  onClick={handleClearChat}
                  disabled={state.loading}
                  className="glass-btn"
                  shape="circle"
                />
              </Tooltip>
              <Tooltip title={t.logout}>
                <Button
                  icon={<LogoutOutlined />}
                  onClick={handleLogout}
                  className="glass-btn"
                  shape="circle"
                />
              </Tooltip>
            </>
          )}

          {/* 移动端只显示必要按钮，减少拥挤 */}
          {isMobile && (
            <>
              <Button
                icon={<GlobalOutlined />}
                onClick={actions.toggleLanguage}
                className="glass-btn"
                shape="circle"
                size="small"
                style={{ color: '#94a3b8' }}
              />
              <Tooltip title={`User: ${state.userId?.slice(0, 4)}...`}>
                <Tag icon={<UserOutlined />} color="blue" style={{
                  margin: 0,
                  padding: '4px 8px',
                  borderRadius: '12px',
                  fontSize: '11px',
                  background: 'rgba(14, 165, 233, 0.1)',
                  border: '1px solid rgba(14, 165, 233, 0.2)',
                  color: '#0ea5e9'
                }}>
                  {state.userId?.slice(0, 3)}...
                </Tag>
              </Tooltip>
            </>
          )}
        </Space>
      </Header>

      <Layout style={{ background: 'transparent' }}>
        {/* Left Sidebar: History */}
        <Sider
          width={280}
          collapsed={isMobile ? leftSidebarCollapsed : leftSidebarCollapsed}
          collapsedWidth={0}
          className="glass-panel mobile-sidebar-left"
          style={{
            borderRight: '1px solid rgba(255, 255, 255, 0.05)',
            background: '#0f172a', // slate-900
            display: isMobile ? (leftSidebarCollapsed ? 'none' : 'block') : 'block',
            position: isMobile ? 'absolute' : 'relative',
            left: 0,
            top: 64,
            height: isMobile ? 'calc(100vh - 64px)' : 'auto',
            zIndex: 100
          }}
        >
          {!isMobile && (
            <div style={{
              padding: '12px 16px',
              borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <Text style={{ color: '#94a3b8', fontSize: '12px', fontWeight: 500 }}>
                {t.sessions || 'Sessions'}
              </Text>
              <Button
                type="text"
                size="small"
                icon={leftSidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                onClick={() => setLeftSidebarCollapsed(!leftSidebarCollapsed)}
                style={{ color: '#94a3b8' }}
              />
            </div>
          )}
          <div style={{ padding: isMobile ? '20px' : '20px 20px 20px 0', height: '100%', overflowY: 'auto' }}>
            <SessionPanel />

            {/* 移动端额外功能按钮 */}
            {isMobile && (
              <div style={{ marginTop: '20px', paddingTop: '20px', borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <Tooltip title={t.clearChat}>
                    <Button
                      icon={<ClearOutlined />}
                      onClick={handleClearChat}
                      disabled={state.loading}
                      className="glass-btn"
                      block
                      style={{
                        height: '44px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}
                    >
                      {t.clearChat}
                    </Button>
                  </Tooltip>
                  <Tooltip title={t.logout}>
                    <Button
                      icon={<LogoutOutlined />}
                      onClick={handleLogout}
                      className="glass-btn"
                      block
                      danger
                      style={{
                        height: '44px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}
                    >
                      {t.logout}
                    </Button>
                  </Tooltip>
                </div>
              </div>
            )}
          </div>
        </Sider>

        {/* Desktop Left Sidebar Toggle Button - Always Visible when not collapsed */}
        {!isMobile && (
          <Button
            type="text"
            size="small"
            icon={leftSidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setLeftSidebarCollapsed(!leftSidebarCollapsed)}
            style={{
              position: 'absolute',
              left: leftSidebarCollapsed ? '16px' : '296px',
              top: '80px',
              zIndex: 101,
              backgroundColor: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              color: '#94a3b8',
              borderRadius: '8px',
              width: '32px',
              height: '32px',
              backdropFilter: 'blur(8px)',
              transition: 'all 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          />
        )}

        {/* Left Sidebar Toggle Button (Mobile) */}
        {isMobile && (
          <Button
            type="text"
            size="large"
            icon={<DoubleRightOutlined />}
            onClick={() => setLeftSidebarCollapsed(!leftSidebarCollapsed)}
            style={{
              position: 'absolute',
              left: leftSidebarCollapsed ? '16px' : '296px',
              top: '80px',
              zIndex: 101,
              backgroundColor: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              color: '#94a3b8',
              borderRadius: '8px',
              width: '40px',
              height: '40px',
              backdropFilter: 'blur(8px)',
              transition: 'all 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          />
        )}

        {/* Center: Chat Area */}
        <Content style={{
          display: 'flex',
          position: 'relative',
          background: '#020617', // slate-950
          marginLeft: isMobile ? 0 : (leftSidebarCollapsed ? 0 : 'auto'),
          marginRight: isMobile ? 0 : (rightSidebarCollapsed ? 0 : 'auto'),
          minHeight: '100vh' // 确保内容区域高度固定
        }}>
          <div
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              height: '100vh', // 固定高度防止布局变化
              position: 'relative',
              maxWidth: isMobile ? '100%' : 'none',
              width: isMobile ? '100%' : 'auto'
            }}
            onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
            onDrop={(e) => {
              e.preventDefault(); e.stopPropagation();
              const files = Array.from(e.dataTransfer.files);
              if (files.length > 0) files.forEach(file => handleUpload(file));
            }}
          >
            <div
              ref={chatContainerRef}
              style={{
                flex: 1,
                padding: isMobile ? '16px 16px 60px 16px' : '24px 40px 100px 40px', // 移动端减少底部padding
                overflowY: 'auto',
                backgroundColor: 'transparent',
                maxWidth: '100%',
                minHeight: 0 // 确保flex子项可以收缩
              }}
            >
              {(!state.currentChatSession?.history || state.currentChatSession.history.length === 0) ? (
                <div style={{ 
                  height: '100%', 
                  display: 'flex', 
                  flexDirection: 'column', 
                  justifyContent: 'center', 
                  alignItems: 'center',
                  maxWidth: '800px',
                  margin: '0 auto'
                }}>
                  <Title level={2} style={{ marginBottom: '10px', textAlign: 'center', color: '#e2e8f0' }}>
                    {t.welcomeTitle}
                  </Title>
                  <Text style={{ marginBottom: '40px', textAlign: 'center', color: '#94a3b8', fontSize: '16px' }}>
                    {t.welcomeSubtitle}
                  </Text>
                  
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr',
                    gap: isMobile ? '16px' : '20px',
                    width: '100%'
                  }}>
                    {SHORTCUT_CARDS.map((card, idx) => (
                      <div
                        key={idx}
                        className="glass-card"
                        style={{
                          padding: isMobile ? '20px' : '24px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: isMobile ? '12px' : '16px',
                          background: '#1e293b', // slate-800
                          border: '1px solid #334155' // slate-700
                        }}
                        onClick={() => handleSendMessage(card.prompt)}
                      >
                        <div style={{
                          width: isMobile ? '40px' : '48px',
                          height: isMobile ? '40px' : '48px',
                          borderRadius: '12px',
                          background: 'rgba(255,255,255,0.05)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0
                        }}>
                          <div style={{ fontSize: isMobile ? '20px' : '24px' }}>
                            {card.icon}
                          </div>
                        </div>
                        <div>
                          <Text strong style={{ fontSize: '16px', display: 'block', color: '#e2e8f0' }}>{card.title}</Text>
                          <Text style={{ fontSize: '13px', color: '#94a3b8' }}>{card.desc}</Text>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <>
                  {state.currentChatSession?.history.map((msg, index) => renderMessage(msg, index))}
                  <div ref={messagesEndRef} />
                </>
              )}
            </div>

            {/* Floating Input Area */}
            <div style={{
              position: 'absolute',
              bottom: isMobile ? '10px' : '30px', // 移动端调整为10px
              left: '50%',
              transform: 'translateX(-50%)',
              width: isMobile ? 'calc(100% - 32px)' : '90%',
              maxWidth: isMobile ? '100%' : '800px',
              zIndex: 20,
              padding: isMobile ? '0 16px' : '0'
            }}>
              {state.error && (
                <Alert
                  message={state.error}
                  type="error"
                  showIcon
                  closable
                  style={{ marginBottom: '16px', borderRadius: '12px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', color: '#fca5a5' }}
                />
              )}
              
              <div style={{
                backgroundColor: '#1e293b', // slate-800
                borderRadius: '30px',
                padding: '8px 16px',
                boxShadow: '0 20px 40px rgba(0, 0, 0, 0.4)',
                border: '1px solid #334155', // slate-700
                display: 'flex',
                alignItems: 'flex-end',
                gap: '12px'
              }}>
                <Upload
                  beforeUpload={handleUpload}
                  showUploadList={false}
                  multiple
                  disabled={uploading || state.loading}
                >
                  <Button 
                    icon={<PaperClipOutlined style={{ fontSize: '20px' }} />} 
                    loading={uploading}
                    type="text"
                    style={{ 
                      color: '#94a3b8', 
                      width: '40px', 
                      height: '40px',
                      marginBottom: '2px'
                    }}
                  />
                </Upload>
                
                <TextArea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  onPaste={(e) => {
                    const items = e.clipboardData.items;
                    for (let i = 0; i < items.length; i++) {
                      if (items[i].kind === 'file') {
                        const file = items[i].getAsFile();
                        if (file) handleUpload(file);
                      }
                    }
                  }}
                  placeholder={t.inputPlaceholder}
                  autoSize={{ minRows: 1, maxRows: 6 }}
                  variant="borderless"
                  style={{ 
                    flex: 1, 
                    resize: 'none', 
                    padding: '10px 0',
                    fontSize: '16px',
                    color: '#f1f5f9',
                    background: 'transparent'
                  }}
                  disabled={state.responding || state.loading}
                />
                
                <Button
                  type="primary"
                  icon={<SendOutlined />}
                  onClick={() => handleSendMessage()}
                  loading={state.responding || state.loading}
                  disabled={!inputValue.trim()}
                  className="send-btn"
                  style={{ marginBottom: '4px' }}
                />
              </div>
            </div>
          </div>

          <Sider
            width={340}
            collapsed={isMobile ? rightSidebarCollapsed : rightSidebarCollapsed}
            collapsedWidth={0}
            theme="light"
            className="mobile-sidebar-right"
            style={{
              borderLeft: '1px solid rgba(51, 65, 85, 0.5)', // border-slate-700
              background: '#0f172a', // slate-900
              padding: '16px', // p-4
              display: isMobile ? (rightSidebarCollapsed ? 'none' : 'block') : 'block',
              position: isMobile ? 'absolute' : 'relative',
              right: 0,
              top: 64,
              height: isMobile ? 'calc(100vh - 64px)' : 'auto',
              zIndex: 100
            }}
          >
            {!isMobile && (
              <div style={{
                padding: '12px 16px',
                borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '16px'
              }}>
                <Text style={{ color: '#94a3b8', fontSize: '12px', fontWeight: 500 }}>
                  {t.tools || 'Tools'}
                </Text>
                <Button
                  type="text"
                  size="small"
                  icon={rightSidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                  onClick={() => setRightSidebarCollapsed(!rightSidebarCollapsed)}
                  style={{ color: '#94a3b8' }}
                />
              </div>
            )}
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              {/* Segmented Control */}
              <div style={{
                width: '100%',
                backgroundColor: '#1e293b', // slate-800
                padding: '4px',
                borderRadius: '8px',
                display: 'flex',
                marginBottom: '24px'
              }}>
                <div
                  onClick={() => setActiveTab('params')}
                  style={{
                    flex: 1,
                    textAlign: 'center',
                    padding: '6px 0',
                    borderRadius: '6px',
                    fontSize: '13px',
                    fontWeight: 500,
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    backgroundColor: activeTab === 'params' ? '#2563eb' : 'transparent', // blue-600
                    color: activeTab === 'params' ? 'white' : '#94a3b8', // slate-400
                    boxShadow: activeTab === 'params' ? '0 1px 2px 0 rgba(0, 0, 0, 0.05)' : 'none'
                  }}
                >
                  {t.parameters}
                </div>
                <div
                  onClick={() => setActiveTab('files')}
                  style={{
                    flex: 1,
                    textAlign: 'center',
                    padding: '6px 0',
                    borderRadius: '6px',
                    fontSize: '13px',
                    fontWeight: 500,
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    backgroundColor: activeTab === 'files' ? '#2563eb' : 'transparent',
                    color: activeTab === 'files' ? 'white' : '#94a3b8',
                    boxShadow: activeTab === 'files' ? '0 1px 2px 0 rgba(0, 0, 0, 0.05)' : 'none'
                  }}
                >
                  {t.files}
                </div>
              </div>

              <div style={{ flex: 1, overflowY: 'auto' }}>
                {activeTab === 'params' ? <ParamPanel /> : <FilePanel />}
              </div>
            </div>
          </Sider>

        {/* Desktop Right Sidebar Toggle Button - Always Visible when not collapsed */}
        {!isMobile && (
          <Button
            type="text"
            size="small"
            icon={rightSidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setRightSidebarCollapsed(!rightSidebarCollapsed)}
            style={{
              position: 'absolute',
              right: rightSidebarCollapsed ? '16px' : '356px',
              top: '80px',
              zIndex: 101,
              backgroundColor: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              color: '#94a3b8',
              borderRadius: '8px',
              width: '32px',
              height: '32px',
              backdropFilter: 'blur(8px)',
              transition: 'all 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          />
        )}

        {/* Right Sidebar Toggle Button (Mobile) */}
        {isMobile && (
          <Button
            type="text"
            size="large"
            icon={<DoubleLeftOutlined />}
            onClick={() => setRightSidebarCollapsed(!rightSidebarCollapsed)}
            style={{
              position: 'absolute',
              right: rightSidebarCollapsed ? '16px' : '356px',
              top: '80px',
              zIndex: 101,
              backgroundColor: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              color: '#94a3b8',
              borderRadius: '8px',
              width: '40px',
              height: '40px',
              backdropFilter: 'blur(8px)',
              transition: 'all 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          />
        )}
        </Content>
      </Layout>
    </Layout>
  );
}

export default Chat;