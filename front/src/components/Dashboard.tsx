import React, { useEffect, useMemo, useState, useCallback, useRef } from 'react';
import { toast } from 'sonner';
import { GlassButton } from './GlassButton';
import { ThemeToggle } from './ThemeToggle';
import { AddPatientDialog } from './AddPatientDialog';
import { Card } from './ui/card';
import { Input } from './ui/input';
import {
    Search,
    User as UserIcon,
    FileText,
    Activity,
    Plus,
    Download,
    Play,
    Trash2,
    Loader2,
    Eye
} from 'lucide-react';

import {
    motion,
    AnimatePresence
} from 'motion/react';

import {
    DropdownMenu,
    DropdownMenuTrigger,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuSeparator
} from './ui/dropdown-menu';

import {
    AlertDialog,
    AlertDialogContent,
    AlertDialogHeader,
    AlertDialogTitle,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogCancel,
    AlertDialogAction
} from './ui/alert-dialog';

import {
    Dialog,
    DialogContent,
    DialogTitle
} from './ui/dialog';

import {
    getResearchesForUser,
    createResearch,
    startProcessing,
    deleteResearch,
    getDownloadUrl,
    addClassesAreas,
    getResearchById,
    deletePatient
} from '../api/api';

import { v4 as uuidv4 } from 'uuid';

// --- ИНТЕРФЕЙСЫ ДАННЫХ ---
interface User { id: string; email: string; }
interface ClassesArea {
  id: string;
  area: number;
  localization: { zone_name: string };
  classification: { class_name: string };
}

interface Research {
  id: string;
  name: string;
  date: string;
  status: string;
  classes_areas: ClassesArea[];
}

interface PatientGroup {
  patientId: string;
  patientCode: string;
  researches: Research[];
  lastUpdate: string;
}

interface DashboardProps {
    user: User;
    onLogout: () => void;
    isDark: boolean;
    onThemeToggle: () => void;
}

type FileStatus = 'Загружен' | 'В очереди' | 'В обработке' | 'Готов';

export function Dashboard({ user, onLogout, isDark, onThemeToggle }: DashboardProps) {
  // --- СОСТОЯНИЯ ---
  const [patientGroups, setPatientGroups] = useState<PatientGroup[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isAddPatientOpen, setIsAddPatientOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState('10');
  const [sortField, setSortField] = useState<'date' | 'name'>('date');
  const [sortDir, setSortDir] = useState<'desc' | 'asc'>('desc');
  const [selectedTab, setSelectedTab] = useState<'all' | FileStatus>('all');
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [deleteDialog, setDeleteDialog] = useState<{ open: boolean; researchId: string | null }>({ open: false, researchId: null });
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  type JsonDialogState = { open: boolean; title: string; data: unknown | null };
  const [jsonDialog, setJsonDialog] = useState<JsonDialogState>({ open: false, title: '', data: null });
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const segRef = useRef<HTMLDivElement | null>(null);
  const [pillRect, setPillRect] = useState<{ left: number; width: number; }>({ left: 0, width: 0 });
  const [deletePatientDialog, setDeletePatientDialog] = useState<{ open: boolean; patientId: string | null; patientCode: string | null }>({ open: false, patientId: null, patientCode: null });
  const [downloadDialog, setDownloadDialog] = useState<{ open: boolean; research: Research | null }>({ open: false, research: null });

  // --- ФУНКЦИИ API ---
  const fetchResearches = useCallback(async () => {
    setIsLoading(true);
    try {
      const apiResearches = await getResearchesForUser(user.id);

      // --- ЛОГИКА ГРУППИРОВКИ ---
      const groups: Record<string, PatientGroup> = {};

      for (const res of apiResearches) {
        const patientId = res.patient.id;
        if (!groups[patientId]) {
          groups[patientId] = {
            patientId: patientId,
            patientCode: res.patient.code,
            researches: [],
            lastUpdate: res.upload_date,
          };
        }
        groups[patientId].researches.push({
          id: res.id,
          name: res.name,
          date: new Date(res.upload_date).toLocaleString('ru-RU'),
          status: res.status.status_name,
          classes_areas: res.classes_areas,
        });
        if (res.upload_date > groups[patientId].lastUpdate) {
            groups[patientId].lastUpdate = res.upload_date;
        }
      }

      const groupedArray = Object.values(groups);
      setPatientGroups(groupedArray);

    } catch (error: any) {
      toast.error(`Не удалось загрузить данные: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [user.id]);

  useEffect(() => {
    fetchResearches();
  }, [fetchResearches]);


  const handleAddResearch = async (data: { patientId: string | null; file: File | null }) => {
    if (!data.file) {
      toast.error('Выберите zip-архив для загрузки.');
      return;
    }
    const patientId = data.patientId || uuidv4();
    const formData = new FormData();
    formData.append('user_email', user.email);
    formData.append('patient_id', patientId);
    formData.append('file', data.file);
    try {
      setIsLoading(true);
      await createResearch(formData);
      toast.success(`Исследование для пациента ${patientId} создано.`);
      await fetchResearches();
    } catch (error: any) {
      toast.error(`Ошибка создания исследования: ${error.message}`);
    } finally {
      setIsLoading(false);
      setIsAddPatientOpen(false);
    }
  };

  const handleAnalyze = async (researchId: string) => {
    setProcessingId(researchId);
    try {
      await startProcessing(researchId);
      toast.info(`Исследование ${researchId.substring(0, 5)} отправлено на обработку.`);
       setPatientGroups(prevGroups =>
        prevGroups.map(group => ({
          ...group,
          researches: group.researches.map(research =>
            research.id === researchId
              ? { ...research, status: 'В очереди' }
              : research
          )
        }))
      );
      setTimeout(fetchResearches, 10000);
    } catch (error: any) {
      toast.error(`Ошибка запуска обработки: ${error.message}`);
    } finally {
      setProcessingId(null);
    }
  };


  const handleDeleteConfirmed = async () => {
    const researchId = deleteDialog.researchId;
    if (!researchId) return;
    try {
      await deleteResearch(researchId);
      toast.success(`Исследование ${researchId.substring(0, 5)} удалено.`);
      setDeleteDialog({ open: false, researchId: null });
      fetchResearches();
    } catch (error: any) {
      toast.error(`Ошибка удаления: ${error.message}`);
    }
  };

  const handleViewJson = async (researchId: string) => {
      try {
        const researchData = await getResearchById(researchId);
        setJsonDialog({ open: true, title: `Результаты: ${researchData.name}`, data: researchData.classes_areas });
      } catch(error: any) {
        toast.error(`Не удалось загрузить детали: ${error.message}`)
      }
  };

  const handleDeletePatientConfirmed = async () => {
        const patientId = deletePatientDialog.patientId;
        if (!patientId) return;

        try {
            await deletePatient(patientId);
            toast.success(`Пациент ${deletePatientDialog.patientCode} и все его исследования удалены.`);
            fetchResearches(); // Обновляем список
        } catch (error: any) {
            toast.error(`Ошибка удаления пациента: ${error.message}`);
        } finally {
            setDeletePatientDialog({ open: false, patientId: null, patientCode: null });
        }
    };

  // --- Логика UI ---

  const toggleRow = (patientId: string) => setExpanded((prev) => ({ ...prev, [patientId]: !prev[patientId] }));
  const handleSort = (field: 'lastUpdate' | 'patientCode') => {
    setSortField(field);
    setSortDir(prev => (prev === 'asc' ? 'desc' : 'asc'));
  };
  const renderSortIndicator = (field: 'lastUpdate' | 'patientCode') => {
    if (sortField !== field) return '↕';
    return sortDir === 'asc' ? '▲' : '▼';
  };

  // --- ФИЛЬТРАЦИЯ И ПАГИНАЦИЯ ---
  useEffect(() => {
    const timerId = setTimeout(() => {
      setDebouncedQuery(searchQuery);
      setCurrentPage(1);
    }, 300);
    return () => clearTimeout(timerId);
  }, [searchQuery]);

  const filteredGroups = useMemo(() => {
    const statusFiltered = selectedTab === 'all'
      ? patientGroups
      : patientGroups.filter(group => group.researches.some(r => r.status === selectedTab));

    const q = debouncedQuery.toLowerCase();
    if (!q) {
      return statusFiltered;
    }
    return statusFiltered.filter(group =>
      group.patientCode.toLowerCase().includes(q) ||
      group.researches.some(r => r.name.toLowerCase().includes(q))
    );
  }, [patientGroups, debouncedQuery, selectedTab]);

  const sortedGroups = useMemo(() => {
    const arr = [...filteredGroups];
    if (!sortField) return arr;
    arr.sort((a, b) => {
      const aVal = sortField === 'lastUpdate' ? a.lastUpdate : a.patientCode;
      const bVal = sortField === 'lastUpdate' ? b.lastUpdate : b.patientCode;
      if (aVal < bVal) return sortDir === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });
    return arr;
  }, [filteredGroups, sortField, sortDir]);

  const perPage = Math.max(1, parseInt(itemsPerPage, 10) || 10);
  const totalItems = sortedGroups.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / perPage));
  const pagedGroups = useMemo(() => {
    const start = (currentPage - 1) * perPage;
    return sortedGroups.slice(start, start + perPage);
  }, [sortedGroups, currentPage, perPage]);

  const totalResearchesCount = useMemo(() => {
    return patientGroups.reduce((acc, group) => acc + group.researches.length, 0);
  }, [patientGroups]);

  const completedResearchesCount = useMemo(() => {
    return patientGroups.reduce((acc, group) => {
      return acc + group.researches.filter(r => r.status === 'Готов').length;
    }, 0);
  }, [patientGroups]);

  useEffect(() => {
    const el = tabRefs.current[selectedTab];
    if (el) {
      setPillRect({ left: el.offsetLeft, width: el.offsetWidth });
    }
  }, [selectedTab, pagedGroups]);


   const statusCounts = useMemo(() => {
    const counts: Record<FileStatus, number> = {
      'Загружен': 0,
      'В очереди': 0,
      'В обработке': 0,
      'Готов': 0,
    };
    for (const group of patientGroups) {
      for (const research of group.researches) {
        if (research.status in counts) {
          counts[research.status as FileStatus]++;
        }
      }
    }
    return counts;
  }, [patientGroups]);

  return (
    <div className={`min-h-screen transition-colors duration-500 relative ${
        isDark
        ? 'bg-slate-900 text-white'
        : 'bg-slate-100 text-slate-900'
    }`}>
      <div className="relative z-10 p-6 max-w-7xl mx-auto">
        <header className="flex justify-between items-center mb-8">
          <div className="flex items-center gap-3">
            <img src="/logo.gif" alt="Логотип" className="h-12 w-12 object-contain rounded-md shadow-sm" />
            <h1 className={`text-2xl font-bold ${
                isDark
                ? 'text-white'
                : 'text-slate-800'
            }`}>
              Медицинская система МРТ
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <ThemeToggle isDark={isDark} onToggle={onThemeToggle} />
            <span className={`text-sm ${
                isDark
                ? 'text-slate-300'
                : 'text-slate-700'
            }`}>{user.email}</span>
            <GlassButton onClick={onLogout} variant="ghost">Выйти</GlassButton>
          </div>
        </header>

        {/* Статистика */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <Card className={`p-6 backdrop-blur-sm ${
                isDark
                ? 'bg-slate-800/50 border-slate-700/50'
                : 'bg-white/50 border-slate-200/50'
            }`}>
                <div className="flex items-center gap-3">
                    <UserIcon className="w-6 h-6 text-blue-400" />
                    <div>
                        <p className="text-sm text-slate-400">Пациентов</p>
                        <p className="text-2xl font-bold">{patientGroups.length}</p>
                    </div>
                </div>
            </Card>
            <Card className={`p-6 backdrop-blur-sm ${
                isDark
                ? 'bg-slate-800/50 border-slate-700/50'
                : 'bg-white/50 border-slate-200/50'
            }`}>
                <div className="flex items-center gap-3">
                    <FileText className="w-6 h-6 text-green-400" />
                    <div>
                        <p className="text-sm text-slate-400">Всего исследований</p>
                        <p className="text-2xl font-bold">{totalResearchesCount}</p>
                    </div>
                </div>
            </Card>
            <Card className={`p-6 backdrop-blur-sm ${
                isDark
                ? 'bg-slate-800/50 border-slate-700/50'
                : 'bg-white/50 border-slate-200/50'
            }`}>
                <div className="flex items-center gap-3">
                    <Activity className="w-6 h-6 text-purple-400" />
                    <div>
                        <p className="text-sm text-slate-400">Готово к просмотру</p>
                        <p className="text-2xl font-bold">{completedResearchesCount}</p>
                    </div>
                </div>
            </Card>
        </div>

        {/* Поиск и Действия */}
        <div className="flex flex-col lg:flex-row gap-4 mb-6 items-center">
          <div className="relative flex-1 w-full">
            <Search className={`absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 ${
                isDark
                ? 'text-slate-400'
                : 'text-slate-500'
            }`} />
            <Input
              placeholder="Поиск по коду пациента или имени исследования..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={`pl-10 w-full backdrop-blur-sm ${
                isDark
                ? 'bg-slate-800/50'
                : 'bg-white/50'
              }`}
            />
          </div>
          <GlassButton variant="primary" onClick={() => setIsAddPatientOpen(true)}>
            <Plus className="w-4 h-4 mr-2" /> Добавить исследование
          </GlassButton>
        </div>
        {/*Вкладки фильтрации*/}
        <div
          ref={segRef}
          role="tablist"
          aria-label="Фильтр статуса"
          className="relative flex items-center gap-4 mb-6"
        >
          <motion.div
            aria-hidden="true"
            className={`absolute h-full rounded-full border-2 ${isDark ? 'border-slate-300' : 'border-blue-600'}`}
            animate={{ x: pillRect.left, width: pillRect.width }}
            transition={{ type: 'spring', stiffness: 400, damping: 30 }}
          />

          {['all', 'Загружен', 'В очереди', 'В обработке', 'Готов'].map(tab => {
            const statusKey = tab as 'all' | FileStatus;
            const isActive = selectedTab === statusKey;
            const count = statusKey !== 'all' ? statusCounts[statusKey] : undefined;

            return (
              <button
                key={statusKey}
                ref={el => (tabRefs.current[statusKey] = el)}
                onClick={() => setSelectedTab(statusKey)}
                className={`relative z-10 flex-shrink-0 px-4 py-2 rounded-full text-sm font-medium transition-colors duration-200 focus:outline-none ${
                  isActive
                    ? (isDark ? 'text-white' : 'text-blue-700')
                    : (isDark ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-black')
                }`}
              >
                {statusKey === 'all' ? 'Все' : statusKey}
                {typeof count === 'number' && (
                  <span className={`ml-2 font-normal ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                    {count}
                  </span>
                )}
              </button>
            );
          })}
        </div>
        {/* Table */}
        <Card className={` backdrop-blur-sm overflow-hidden ${
            isDark
            ? 'bg-slate-800/50'
            : 'bg-white/50'
        }`}>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className={
                    isDark
                    ? 'border-b border-slate-700'
                    : 'border-b border-slate-200'
                }>
                    <th className="p-4 font-semibold">ID Пациента</th>
                    <th className="p-4"><button onClick={() => handleSort('date')} className="font-semibold flex items-center gap-2">Дата обновления {renderSortIndicator('date')}</button></th>
                    <th className="p-4 font-semibold text-center">Исследований</th>
                    <th className="p-4 text-center font-semibold">Действия</th>
                </tr>
              </thead>
              <tbody>
                {isLoading ? (
                  <tr>
                    <td colSpan={4} className="p-8 text-center">
                        <Loader2 className="w-8 h-8 animate-spin mx-auto text-blue-500" />
                    </td>
                  </tr>
                ) : pagedGroups.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="p-8 text-center text-slate-500">Пациенты не найдены.</td>
                  </tr>
                ) : (
                  pagedGroups.map((group) => (
                    <React.Fragment key={group.patientId}>
                      <tr className={`border-t hover:bg-slate-500/10 ${
                        isDark
                        ? 'border-slate-800'
                        : 'border-slate-100'
                      }`}>
                        <td className="p-4 font-medium">{group.patientCode}</td>
                        <td className="p-4 text-sm text-slate-400">{new Date(group.lastUpdate).toLocaleString('ru-RU')}</td>
                        <td className="p-4 text-center font-mono">{group.researches.length}</td>
                        <td className="p-4 text-center">
                           <GlassButton size="sm" variant={expanded[group.patientId] ? 'primary' : 'ghost'} onClick={() => toggleRow(group.patientId)}>
                            {expanded[group.patientId]
                                ? 'Скрыть'
                                : 'Показать'
                            }
                           </GlassButton>
                        </td>
                      </tr>
                      <tr>
                        <td colSpan={4} className="p-0 border-none">
                          <AnimatePresence initial={false}>
                            {expanded[group.patientId] && (
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.3, ease: "easeInOut" }}
                                className={`${isDark ? 'bg-slate-800/40' : 'bg-slate-50'} overflow-hidden`}
                              >
                                <div className="p-4 space-y-4">
                                  <div className="px-4 pt-2 flex justify-between items-center">
                                      <div>
                                        <h4 className="font-semibold text-sm mb-1">ID Пациента</h4>
                                        <Input
                                          readOnly
                                          value={group.patientId}
                                          className="font-mono text-xs bg-slate-700/50 border-slate-600 cursor-copy"
                                          onClick={(e) => {
                                              navigator.clipboard.writeText(group.patientId);
                                              toast.success("ID пациента скопирован в буфер обмена");
                                        }}/>
                                      </div>
                                      <GlassButton
                                        size="sm"
                                        variant="ghost"
                                        onClick={() => setDeletePatientDialog({ open: true, patientId: group.patientId, patientCode: group.patientCode })}
                                        className="text-red-500 hover:bg-red-500/10"
                                      >
                                        <Trash2 className="w-4 h-4 mr-2" /> Удалить пациента
                                      </GlassButton>
                                  </div>
                                  <div className="px-4 font-semibold">Папки исследований</div>
                                  {group.researches.map(research => (
                                    <div key={research.id} className={`mx-4 flex items-center justify-between rounded-md border px-4 py-3 ${isDark ? 'bg-slate-700/50 border-slate-600' : 'bg-slate-100 border-slate-200'}`}>
                                        <div>
                                            <div className='font-medium'>{research.name}</div>
                                            <div className='text-xs text-slate-400'>Статус: {research.status}</div>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <GlassButton size="sm" variant="ghost" onClick={() => handleAnalyze(research.id)} disabled={processingId === research.id || research.status !== 'Загружен'} title="Начать анализ">
                                                {processingId === research.id ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 mr-1" />} Анализ
                                            </GlassButton>
                                            <GlassButton size="sm" variant="ghost" onClick={() => handleViewJson(research.id)} disabled={research.status !== 'Готов'} title="Посмотреть результаты (JSON)">
                                                <Eye className="w-4 h-4 mr-1" /> Результаты
                                            </GlassButton>
                                            <GlassButton
                                                size="sm"
                                                variant="ghost"
                                                disabled={research.status !== 'Готов'}
                                                title="Скачать файлы"
                                                onClick={() => setDownloadDialog({ open: true, research: research })}
                                            >
                                                <Download className="w-4 h-4 mr-1" /> Скачать
                                            </GlassButton>

                                            <GlassButton
                                                size="sm"
                                                variant="ghost"
                                                onClick={() => setDeleteDialog({ open: true, researchId: research.id })}
                                                title="Удалить исследование"
                                                className="text-red-500 hover:bg-red-500/10"
                                            >
                                                <Trash2 className="w-4 h-4 mr-1" /> Удалить
                                            </GlassButton>
                                        </div>
                                    </div>
                                  ))}
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </td>
                      </tr>
                    </React.Fragment>
                  ))
                )}
              </tbody>
            </table>
          </div>
          {/* Пагинация */}
          <div className={`flex justify-between items-center p-4 border-t ${isDark ? 'border-slate-700/50' : 'border-slate-200/50'}`}>
              <span className={`text-sm ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                {totalItems === 0 ? 0 : (currentPage - 1) * perPage + 1}-{Math.min(currentPage * perPage, totalItems)} из {totalItems}
              </span>
              <div className="flex items-center gap-3">
                <span className={`text-sm ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>Стр. {currentPage} / {totalPages}</span>
                <div className="flex gap-2">
                  <GlassButton size="sm" variant="ghost" onClick={() => currentPage > 1 && setCurrentPage(currentPage - 1)} disabled={currentPage === 1}>Назад</GlassButton>
                  <GlassButton size="sm" variant="ghost" onClick={() => currentPage < totalPages && setCurrentPage(currentPage + 1)} disabled={currentPage === totalPages}>Вперёд</GlassButton>
                </div>
              </div>
          </div>
        </Card>
      </div>

      {/* Dialogs */}
      <AddPatientDialog isOpen={isAddPatientOpen} onClose={() => setIsAddPatientOpen(false)} onAddPatient={handleAddResearch} isDark={isDark} />

      <AlertDialog open={deleteDialog.open} onOpenChange={(o) => { if (!o) setDeleteDialog({ open: false, researchId: null }); }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Удалить исследование?</AlertDialogTitle>
            <AlertDialogDescription>Это действие нельзя отменить.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setDeleteDialog({ open: false, researchId: null })}>Отмена</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteConfirmed}>Удалить</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={deletePatientDialog.open} onOpenChange={(o) => { if (!o) setDeletePatientDialog({ open: false, patientId: null, patientCode: null }); }}>
        <AlertDialogContent className={`backdrop-blur-xl border-none shadow-2xl ${isDark ? 'bg-slate-900/80 text-slate-200' : 'bg-white/80 text-slate-800'}`}>
            <AlertDialogHeader>
                <AlertDialogTitle>Вы уверены?</AlertDialogTitle>
                <AlertDialogDescription className={isDark ? 'text-slate-400' : 'text-slate-600'}>
                    Вы собираетесь удалить пациента <strong className="font-bold">{deletePatientDialog.patientCode}</strong>.
                    Это действие навсегда удалит самого пациента и **все связанные с ним исследования**. Отменить это действие будет невозможно.
                </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
                <AlertDialogCancel onClick={() => setDeletePatientDialog({ open: false, patientId: null, patientCode: null })}>Отмена</AlertDialogCancel>
                <AlertDialogAction onClick={handleDeletePatientConfirmed} className="bg-red-600 hover:bg-red-700 text-white">Удалить</AlertDialogAction>
            </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

        <Dialog open={jsonDialog.open} onOpenChange={(o) => setJsonDialog((s) => ({ ...s, open: o }))}>
        <DialogContent className={`sm:max-w-3xl w-[min(100%,1100px)]
             ${isDark
                ? 'bg-slate-900 border-slate-700'
                : 'bg-white border-slate-200'
             }`}>
          <DialogTitle>{jsonDialog.title || 'Результаты анализа'}</DialogTitle>
          <div className={`max-h-[70vh] overflow-auto rounded-md border backdrop-blur-xl shadow-sm ${isDark ? 'bg-slate-800/80 border-slate-700/70' : 'bg-white/80 border-slate-200/70'}`}>
            {/* Проверяем, что данные являются массивом */}
            {Array.isArray(jsonDialog.data) && jsonDialog.data.length > 0 ? (
              <table className={`w-full text-sm`}>
                <thead className={isDark ? 'bg-slate-800/70' : 'bg-slate-50'}>
                  <tr>
                    <th className="text-left px-4 py-2 w-1/3">Локализация (Localization)</th>
                    <th className="text-left px-4 py-2 w-1/3">Классификация (Classification)</th>
                    <th className="text-right px-4 py-2">Площадь (Area)</th>
                  </tr>
                </thead>
                <tbody>
                  {/* Итерируемся по массиву classes_areas */}
                  {(jsonDialog.data as ClassesArea[]).map((areaItem) => (
                    <tr key={areaItem.id} className={isDark ? 'border-t border-slate-700/60' : 'border-t border-slate-200'}>
                      <td className="px-4 py-2 align-top font-medium">{areaItem.localization.zone_name}</td>
                      <td className="px-4 py-2 align-top">{areaItem.classification.class_name}</td>
                      <td className="px-4 py-2 align-top text-right font-mono">{areaItem.area.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className={`p-4 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>Данные о результатах отсутствуют.</div>
            )}
          </div>
        </DialogContent>
      </Dialog>
      <Dialog open={downloadDialog.open} onOpenChange={(isOpen) => setDownloadDialog({ open: isOpen, research: isOpen ? downloadDialog.research : null })}>
          <DialogContent className={`backdrop-blur-xl border-none shadow-2xl ${isDark ? 'bg-slate-900/80 text-slate-200' : 'bg-white/80 text-slate-800'}`}>
              <DialogTitle>Скачать файлы для: {downloadDialog.research?.name}</DialogTitle>
              <div className="grid grid-cols-2 gap-4 pt-4">
                  <GlassButton variant="secondary" onClick={() => window.open(getDownloadUrl(downloadDialog.research!.id, 't1'))}>
                      <FileText className="w-4 h-4 mr-2" /> T1
                  </GlassButton>
                  <GlassButton variant="secondary" onClick={() => window.open(getDownloadUrl(downloadDialog.research!.id, 't2'))}>
                      <FileText className="w-4 h-4 mr-2" /> T2
                  </GlassButton>
                  <GlassButton variant="secondary" onClick={() => window.open(getDownloadUrl(downloadDialog.research!.id, 'prediction'))}>
                      <FileText className="w-4 h-4 mr-2" /> Prediction
                  </GlassButton>
                  <GlassButton variant="secondary" onClick={() => window.open(getDownloadUrl(downloadDialog.research!.id, 'parcellation'))}>
                      <FileText className="w-4 h-4 mr-2" /> Parcellation
                  </GlassButton>
              </div>
          </DialogContent>
      </Dialog>
    </div>
  );
}