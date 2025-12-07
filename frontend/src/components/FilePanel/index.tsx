import { useState, useEffect } from 'react';
import {
  Button,
  List,
  Typography,
  message,
  Tooltip,
  Popconfirm,
  Spin,
  Card,
  Collapse,
  Badge
} from 'antd';
import {
  DownloadOutlined,
  DeleteOutlined,
  ReloadOutlined,
  FileOutlined,
  FileImageOutlined,
  CodeOutlined,
  FileTextOutlined,
  ExperimentOutlined
} from '@ant-design/icons';

import { useApp } from '../../contexts/AppContext';
import type { FileInfo } from '../../types';

const { Title, Text } = Typography;
const { Panel } = Collapse;

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getFileIcon(filename: string) {
  const ext = filename.split('.').pop()?.toLowerCase();
  if (['png', 'jpg', 'jpeg', 'gif'].includes(ext || '')) return <FileImageOutlined style={{ color: '#fa8c16' }} />;
  if (['json', 'yaml', 'yml', 'toml'].includes(ext || '')) return <CodeOutlined style={{ color: '#52c41a' }} />;
  if (['vasp', 'cif', 'xyz'].includes(ext || '') || filename.includes('POSCAR') || filename.includes('CONTCAR')) return <ExperimentOutlined style={{ color: '#722ed1' }} />;
  return <FileTextOutlined style={{ color: '#1677ff' }} />;
}

function FilePanel() {
  const { state, actions } = useApp();
  
  // Auto-refresh every 5 seconds
  useEffect(() => {
    if (!state.userId) return;
    
    const timer = setInterval(() => {
      actions.loadFiles().catch(err => console.error("Auto-refresh failed", err));
    }, 5000);
    
    return () => clearInterval(timer);
  }, [state.userId, actions]);

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
    message.info('删除功能尚未实现，请联系管理员');
  };

  const handleRefresh = async () => {
    if (!state.userId) return;
    try {
      await actions.loadFiles();
      message.success('文件列表已刷新');
    } catch (error) {
      message.error('刷新文件列表失败');
    }
  };

  // Group files
  const groupedFiles = {
    structures: [] as FileInfo[],
    images: [] as FileInfo[],
    configs: [] as FileInfo[],
    others: [] as FileInfo[]
  };

  (state.files || []).forEach(file => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (['vasp', 'cif', 'xyz'].includes(ext || '') || file.name.includes('POSCAR') || file.name.includes('CONTCAR')) {
      groupedFiles.structures.push(file);
    } else if (['png', 'jpg', 'jpeg', 'gif'].includes(ext || '')) {
      groupedFiles.images.push(file);
    } else if (['json', 'yaml', 'yml', 'toml'].includes(ext || '')) {
      groupedFiles.configs.push(file);
    } else {
      groupedFiles.others.push(file);
    }
  });

  // Sort by modification time (if available) or name
  const sortFiles = (files: FileInfo[]) => files.sort((a, b) => b.name.localeCompare(a.name)); // Simple name sort for now as we don't have mtime in FileInfo yet

  const renderFileList = (files: FileInfo[]) => (
    <List
      size="small"
      dataSource={sortFiles(files)}
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
            avatar={getFileIcon(file.name)}
            title={
              <Text style={{ fontSize: '13px' }} title={file.name} ellipsis>
                {file.name}
              </Text>
            }
            description={
              <Text type="secondary" style={{ fontSize: '11px' }}>
                {formatFileSize(file.size)}
              </Text>
            }
          />
        </List.Item>
      )}
    />
  );

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', padding: '0 4px' }}>
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

      <Text type="secondary" style={{ fontSize: '12px', display: 'block', marginBottom: '12px', padding: '0 4px' }}>
        工作目录: workspace/{state.userId}/files
      </Text>

      <div style={{ flex: 1, overflowY: 'auto' }}>
        {state.loading && !state.files ? (
          <div style={{ textAlign: 'center', padding: '20px' }}>
            <Spin />
          </div>
        ) : (state.files && state.files.length > 0) ? (
          <Collapse defaultActiveKey={['structures', 'images', 'configs', 'others']} ghost size="small">
            {groupedFiles.structures.length > 0 && (
              <Panel header={<span style={{fontWeight: 'bold'}}>⚛️ 结构文件 ({groupedFiles.structures.length})</span>} key="structures">
                {renderFileList(groupedFiles.structures)}
              </Panel>
            )}
            {groupedFiles.images.length > 0 && (
              <Panel header={<span style={{fontWeight: 'bold'}}>🖼️ 图像结果 ({groupedFiles.images.length})</span>} key="images">
                {renderFileList(groupedFiles.images)}
              </Panel>
            )}
            {groupedFiles.configs.length > 0 && (
              <Panel header={<span style={{fontWeight: 'bold'}}>⚙️ 配置文件 ({groupedFiles.configs.length})</span>} key="configs">
                {renderFileList(groupedFiles.configs)}
              </Panel>
            )}
            {groupedFiles.others.length > 0 && (
              <Panel header={<span style={{fontWeight: 'bold'}}>📄 其他文件 ({groupedFiles.others.length})</span>} key="others">
                {renderFileList(groupedFiles.others)}
              </Panel>
            )}
          </Collapse>
        ) : (
          <div style={{ padding: '40px 20px', textAlign: 'center', color: '#8c8c8c' }}>
            <FileOutlined style={{ fontSize: '32px', marginBottom: '16px', color: '#d9d9d9' }} />
            <div>暂无文件</div>
            <div style={{ fontSize: '12px', marginTop: '8px' }}>上传或生成的文件将显示在这里</div>
          </div>
        )}
      </div>
    </div>
  );
}

export default FilePanel;