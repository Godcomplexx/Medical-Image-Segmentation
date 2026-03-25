import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { GlassButton } from './GlassButton';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card } from './ui/card';
import { X, Upload, FileText, Trash2 } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';

interface AddPatientDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onAddPatient: (data: { patientId: string | null; file: File | null }) => void;
  isDark: boolean;
}

export function AddPatientDialog({ isOpen, onClose, onAddPatient, isDark }: AddPatientDialogProps) {
  const [patientId, setPatientId] = useState('');
  const [file, setFile] = useState<File | null>(null);

  // Генерируем плейсхолдер для поля ввода
  const generatePlaceholderId = () => uuidv4();

  // Обработчик выбора файла
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const selectedFile = event.target.files[0];
      // Принимаем только zip-архивы
      if (selectedFile.name.endsWith('.zip')) {
        setFile(selectedFile);
      } else {
        alert('Пожалуйста, выберите файл в формате .zip');
        setFile(null);
        event.target.value = '';
      }
    }
  };

  const removeFile = () => {
    setFile(null);
  };

  // Обработчик отправки формы
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onAddPatient({ patientId: patientId || null, file });
    setPatientId('');
    setFile(null);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className={`w-full max-w-2xl backdrop-blur-xl rounded-3xl shadow-2xl border ${
              isDark 
                ? 'bg-slate-800/90 border-slate-700/50' 
                : 'bg-white/90 border-slate-200/50'
            }`}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-6 border-b border-slate-200/20">
              <h2 className={`text-xl font-semibold ${isDark ? 'text-white' : 'text-slate-900'}`}>
                Добавить новое исследование
              </h2>
              <motion.button
                onClick={onClose}
                className={`p-2 rounded-full hover:bg-slate-200/20 transition-colors ${
                  isDark ? 'text-slate-300 hover:text-white' : 'text-slate-600 hover:text-slate-900'
                }`}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <X className="w-5 h-5" />
              </motion.button>
            </div>

            <form onSubmit={handleSubmit} className="p-6 space-y-6">
              <div className="space-y-2">
                <Label 
                  htmlFor="patientId" 
                  className={isDark ? 'text-white' : 'text-slate-900'}
                >
                  ID Пациента (оставьте пустым для создания нового)
                </Label>
                <Input
                  id="patientId"
                  type="text"
                  placeholder={`Например: ${generatePlaceholderId()}`}
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  className={`transition-all duration-300 rounded-xl ${
                    isDark 
                      ? 'bg-slate-700/50 border-slate-600/50 text-white placeholder:text-slate-400' 
                      : 'bg-white/50 border-slate-300/50 text-slate-900 placeholder:text-slate-500'
                  }`}
                />
              </div>

              <div className="space-y-4">
                <Label className={isDark ? 'text-white' : 'text-slate-900'}>
                  Zip-архив с исследованием
                </Label>
                
                {!file ? (
                  <motion.label 
                    className={`cursor-pointer flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-2xl transition-all duration-300 ${
                      isDark 
                        ? 'border-slate-600 hover:border-slate-500 bg-slate-700/20 hover:bg-slate-700/40' 
                        : 'border-slate-300 hover:border-slate-400 bg-slate-50/50 hover:bg-slate-100/50'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Upload className={`w-12 h-12 mb-4 ${isDark ? 'text-slate-400' : 'text-slate-500'}`} />
                    <p className={`text-center ${isDark ? 'text-white' : 'text-slate-900'}`}>
                      Нажмите для выбора zip-архива
                    </p>
                    <input
                      type="file"
                      onChange={handleFileChange}
                      className="hidden"
                      accept=".zip" // Принимаем только .zip
                    />
                  </motion.label>
                ) : (
                  <Card className={`p-4 ${isDark ? 'bg-slate-700/30 border-slate-600/30' : 'bg-white/50 border-slate-200/50'}`}>
                    <div className="space-y-2 max-h-40 overflow-y-auto">
                        <motion.div
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          className={`flex items-center justify-between p-3 rounded-xl ${
                            isDark ? 'bg-slate-600/50' : 'bg-slate-100/50'
                          }`}
                        >
                          <div className="flex items-center gap-3">
                            <FileText className={`w-5 h-5 ${isDark ? 'text-slate-300' : 'text-slate-600'}`} />
                            <div>
                              <p className={`text-sm font-medium ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                {file.name}
                              </p>
                              <p className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                                {formatFileSize(file.size)}
                              </p>
                            </div>
                          </div>
                          <motion.button
                            type="button"
                            onClick={removeFile}
                            className={`p-2 rounded-lg hover:bg-red-500/20 transition-colors ${
                              isDark ? 'text-slate-400 hover:text-red-400' : 'text-slate-500 hover:text-red-500'
                            }`}
                            whileHover={{ scale: 1.1 }}
                            whileTap={{ scale: 0.9 }}
                          >
                            <Trash2 className="w-4 h-4" />
                          </motion.button>
                        </motion.div>
                    </div>
                  </Card>
                )}
              </div>

              <div className="flex gap-3 pt-4">
                <GlassButton 
                  type="button" 
                  onClick={onClose} 
                  variant="ghost" 
                  className="flex-1"
                >
                  Отмена
                </GlassButton>
                <GlassButton 
                  type="submit" 
                  variant="primary" 
                  className="flex-1"
                  disabled={!file}
                >
                  Добавить исследование
                </GlassButton>
              </div>
            </form>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}