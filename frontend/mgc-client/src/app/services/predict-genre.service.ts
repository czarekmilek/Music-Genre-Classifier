import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root',
})
export class PredictGenreService {
  private apiUrl = '';

  constructor(private http: HttpClient) {}
}
