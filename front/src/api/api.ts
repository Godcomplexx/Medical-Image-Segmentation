const API_BASE_URL = '/api';
const TOKEN_KEY = 'access_token';

// === Token storage ===
export const getToken = (): string | null => localStorage.getItem(TOKEN_KEY);
export const setToken = (token: string) => localStorage.setItem(TOKEN_KEY, token);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);
export const logout = () => clearToken();

// Заголовки с JWT (если есть). extra позволяет добавить Content-Type и т.п.
function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const token = getToken();
  return token ? { ...extra, Authorization: `Bearer ${token}` } : { ...extra };
}

async function handleError(response: Response) {
  if (!response.ok) {
    // Истёкший/невалидный токен — сбрасываем, чтобы UI мог потребовать повторный вход.
    if (response.status === 401) {
      clearToken();
    }
    try {
      const errorData = await response.json();
      const message = errorData.detail || `Ошибка ${response.status}`;
      throw new Error(message);
    } catch {
      throw new Error(`Ошибка сервера: ${response.status} ${response.statusText}`);
    }
  }
  return response;
}

export const registerUser = async (email: string, password: string) => {
  const response = await fetch(`${API_BASE_URL}/users/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  await handleError(response);
  return response.json();
};

// login возвращает JWT; сохраняем его и подгружаем профиль пользователя через /users/me.
export const loginUser = async (email: string, password: string) => {
  const response = await fetch(`${API_BASE_URL}/users/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  await handleError(response);
  const { access_token } = await response.json();
  setToken(access_token);

  const meResponse = await fetch(`${API_BASE_URL}/users/me`, { headers: authHeaders() });
  await handleError(meResponse);
  return meResponse.json(); // { id, email }
};

export const getResearchesForUser = async (userId: string) => {
  const response = await fetch(`${API_BASE_URL}/users/${userId}/researches`, {
    headers: authHeaders(),
  });
  await handleError(response);
  return response.json();
};

export const createResearch = async (formData: FormData) => {
  // Content-Type для FormData проставляется браузером автоматически (с boundary).
  const response = await fetch(`${API_BASE_URL}/researches/upload_zip`, {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
  });
  await handleError(response);
  return response.json();
};

export const startProcessing = async (researchId: string) => {
  // Каталог определяется сервером по research_id; тело не требуется.
  const response = await fetch(`${API_BASE_URL}/researches/${researchId}/process`, {
    method: 'POST',
    headers: authHeaders(),
  });
  await handleError(response);
  return response.json();
};

export const deleteResearch = async (researchId: string) => {
  const response = await fetch(`${API_BASE_URL}/researches/${researchId}`, {
    method: 'DELETE',
    headers: authHeaders(),
  });
  if (response.status !== 204) {
    await handleError(response);
  }
  return true;
};

// Скачивание защищённого файла: fetch с токеном → blob → сохранение.
export const downloadProcessedFile = async (researchId: string, fileType: string) => {
  const response = await fetch(`${API_BASE_URL}/researches/${researchId}/download/${fileType}`, {
    headers: authHeaders(),
  });
  await handleError(response);
  const blob = await response.blob();
  const disposition = response.headers.get('Content-Disposition') || '';
  const match = disposition.match(/filename="?([^"]+)"?/);
  const filename = match ? match[1] : `${fileType}.nii`;

  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
};

export const getResearchById = async (researchId: string) => {
  const response = await fetch(`${API_BASE_URL}/researches/${researchId}`, {
    headers: authHeaders(),
  });
  await handleError(response);
  return response.json();
};

export const addClassesAreas = async (researchId: string, data: string[][]) => {
  const response = await fetch(`${API_BASE_URL}/researches/${researchId}/classes_areas`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(data),
  });
  await handleError(response);
  return response.json();
};

export const deletePatient = async (patientId: string) => {
  const response = await fetch(`${API_BASE_URL}/patients/${patientId}`, {
    method: 'DELETE',
    headers: authHeaders(),
  });
  if (response.status !== 204) {
    await handleError(response);
  }
  return true;
};
