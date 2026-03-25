const API_BASE_URL = '/api';

async function handleError(response: Response) {
  if (!response.ok) {
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

export const loginUser = async (email: string, password: string) => {
  const response = await fetch(`${API_BASE_URL}/users/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  await handleError(response);
  return response.json();
};

export const getResearchesForUser = async (userId: string) => {
  const response = await fetch(`${API_BASE_URL}/users/${userId}/researches`);
  await handleError(response);
  return response.json();
};

export const createResearch = async (formData: FormData) => {
  const response = await fetch(`${API_BASE_URL}/researches/upload_zip`, {
    method: 'POST',
    body: formData,
  });
  await handleError(response);
  return response.json();
};

export const startProcessing = async (researchId: string) => {
  const formData = new FormData();
  const folderName = researchId.substring(0, 5);
  formData.append('folder_name', folderName);
  const response = await fetch(`${API_BASE_URL}/researches/${researchId}/process`, {
    method: 'POST',
    body: formData,
  });
  await handleError(response);
  return response.json();
};

export const deleteResearch = async (researchId: string) => {
  const response = await fetch(`${API_BASE_URL}/researches/${researchId}`, {
    method: 'DELETE',
  });
  if (response.status !== 204) {
    await handleError(response);
  }
  return true;
};

export const getDownloadUrl = (researchId: string, fileType: string) => {
  return `${API_BASE_URL}/researches/${researchId}/download/${fileType}`;
};

export const getResearchById = async (researchId: string) => {
  const response = await fetch(`${API_BASE_URL}/researches/${researchId}`);
  await handleError(response);
  return response.json();
};

export const addClassesAreas = async (researchId: string, data: string[][]) => {
  const response = await fetch(`${API_BASE_URL}/researches/${researchId}/bulk_classes_areas`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  await handleError(response);
  return response.json();
};


export const deletePatient = async (patientId: string) => {
  const response = await fetch(`${API_BASE_URL}/patients/${patientId}`, {
    method: 'DELETE',
  });
  if (response.status !== 204) {
    await handleError(response);
  }
  return true;
};