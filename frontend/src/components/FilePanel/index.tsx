import { useState } from 'react';
import {
  Button,
  Upload,
  List,
  Typography,
  message,
  Tooltip,
  Popconfirm,
  Spin,
  Card
} from 'antd';
import {
  DownloadOutlined,
  DeleteOutlined,
  ReloadOutlined,
  FileOutlined,
  InboxOutlined
} from '@ant-design/icons';
import type { UploadProps } from 'antd';

import { useApp } from '../../contexts/AppContext';
import type { FileInfo } from '../../types';

const { Title, Text } = Typography;
const { Dragger } = Upload;

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function FilePanel() {
  const { state, actions } = useApp();
  const [uploading, setUploading] = useState(false);
  // const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload: UploadProps['beforeUpload'] = async (file) => {
    if (!state.userId) {
      message.error('请先登录');
      return false;
    }

    // 检查文件大小 (10MB限制)
    if (file.size > 10 * 1024 * 1024) {
      message.error('文件大小不能超过10MB');
      return false;
    }

    setUploading(true);
    try {
      await actions.uploadFiles([file]);
      message.success('文件上传成功');
    } catch (error) {
      message.error('文件上传失败');
      console.error('Upload error:', error);
    } finally {
      setUploading(false);
    }

    return false; // 阻止默认上传行为
  };

  const handleDownload = (file: FileInfo) => {
    if (!state.userId) return;

    const downloadUrl = `/api/download/${state.userId}/${file.name}`;
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = file.name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDelete = async (_file: FileInfo) => {
    // 注意：后端目前没有删除文件的API，这里只是界面实现
    message.info('删除功能尚未实现，请联系管理员');
  };

  const handleRefresh = async () => {
    if (!state.userId) return;

    try {
      await actions.loadFiles();
    } catch (error) {
      message.error('刷新文件列表失败');
      console.error('Refresh error:', error);
    }
  };

  const uploadProps: UploadProps = {
    name: 'files',
    multiple: true,
    beforeUpload: handleUpload,
    showUploadList: false,
    disabled: uploading || !state.isAuthenticated
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <Title level={5} style={{ margin: 0 }}>
          文件管理
        </Title>
        <Tooltip title="刷新文件列表">
          <Button
            icon={<ReloadOutlined />}
            onClick={handleRefresh}
            size="small"
            loading={state.loading}
          />
        </Tooltip>
      </div>

      <Text type="secondary" style={{ fontSize: '12px', display: 'block', marginBottom: '12px' }}>
        工作目录: /tmp/{state.userId}
      </Text>

      <Card size="small" style={{ marginBottom: '16px' }}>
        <Dragger {...uploadProps} style={{ padding: '16px' }}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined style={{ fontSize: '32px', color: '#1677ff' }} />
          </p>
          <p className="ant-upload-text">
            点击或拖拽文件到此区域上传
          </p>
          <p className="ant-upload-hint">
            支持单个或批量上传，文件大小不超过10MB
          </p>
        </Dragger>
      </Card>

      {state.loading ? (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <Spin />
        </div>
      ) : (
        <Card
          title="文件列表"
          size="small"
          styles={{ body: { padding: '0' } }}
        >
          {state.currentChatSession?.files && state.currentChatSession.files.length > 0 ? (
            <List
              size="small"
              dataSource={state.currentChatSession.files}
              renderItem={(file: FileInfo) => (
                <List.Item
                  style={{ padding: '8px 12px', borderBottom: '1px solid #f0f0f0' }}
                  actions={[
                    <Tooltip title="下载" key="download">
                      <Button
                        type="text"
                        icon={<DownloadOutlined />}
                        onClick={() => handleDownload(file)}
                        size="small"
                      />
                    </Tooltip>,
                    <Popconfirm
                      title="确定要删除这个文件吗？"
                      onConfirm={() => handleDelete(file)}
                      key="delete"
                    >
                      <Tooltip title="删除">
                        <Button
                          type="text"
                          danger
                          icon={<DeleteOutlined />}
                          size="small"
                        />
                      </Tooltip>
                    </Popconfirm>
                  ]}
                >
                  <List.Item.Meta
                    avatar={<FileOutlined style={{ color: '#1677ff' }} />}
                    title={
                      <Text style={{ fontSize: '14px' }} title={file.name}>
                        {file.name.length > 20 ? file.name.substring(0, 17) + '...' : file.name}
                      </Text>
                    }
                    description={
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {formatFileSize(file.size)}
                      </Text>
                    }
                  />
                </List.Item>
              )}
            />
          ) : (
            <div style={{ padding: '20px', textAlign: 'center', color: '#8c8c8c' }}>
              <FileOutlined style={{ fontSize: '24px', marginBottom: '8px' }} />
              <div>暂无文件</div>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

export default FilePanel;