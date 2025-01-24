import { Injectable } from '@angular/core';
import { HttpClient, HttpEvent, HttpRequest } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ProbObject } from '../models/predict-genre-response.model';

@Injectable({
  providedIn: 'root',
})
export class PredictGenreService {
  private apiUrl = 'http://127.0.0.1:8000';

  constructor(private http: HttpClient) {}

  uploadFile(file: File): Observable<HttpEvent<ProbObject>> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    const req = new HttpRequest('POST', `${this.apiUrl}/classify`, formData, {
      reportProgress: true,
      responseType: 'json',
    });
    return this.http.request<ProbObject>(req);
  }
}
