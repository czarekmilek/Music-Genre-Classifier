import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ProbObject } from '../models/predict-genre-response.model';

@Injectable({
  providedIn: 'root',
})
export class PredictGenreService {
  private apiUrl = 'http://127.0.0.1:8000';

  constructor(private http: HttpClient) {}

  uploadFile(file: File): Observable<ProbObject> {
    const formData = new FormData();
    formData.append('file', file, file.name);

    return this.http.post<ProbObject>(`${this.apiUrl}/classify`, formData);
  }
}
