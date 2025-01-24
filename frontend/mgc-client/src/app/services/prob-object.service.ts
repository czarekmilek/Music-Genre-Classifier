import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { ProbObject } from '../models/predict-genre-response.model';

@Injectable({
  providedIn: 'root',
})
export class ProbObjectService {
  private readonly storageKey = 'classificationResult';
  private probObjectSource = new BehaviorSubject<ProbObject | null>(
    this.getFromStorage()
  );
  probObject$ = this.probObjectSource.asObservable();

  setProbObject(probObject: ProbObject) {
    this.probObjectSource.next(probObject);
    this.saveToStorage(probObject);
  }

  getProbObject(): ProbObject | null {
    return this.probObjectSource.getValue();
  }

  private saveToStorage(probObject: ProbObject) {
    sessionStorage.setItem(this.storageKey, JSON.stringify(probObject));
  }

  private getFromStorage(): ProbObject | null {
    const data = sessionStorage.getItem(this.storageKey);
    return data ? JSON.parse(data) : null;
  }

  clearProbObject() {
    this.probObjectSource.next(null);
    sessionStorage.removeItem(this.storageKey);
  }
}
