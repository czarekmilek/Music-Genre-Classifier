import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { ProbObject } from '../models/predict-genre-response.model';

@Injectable({
  providedIn: 'root',
})
export class ProbObjectService {
  private readonly propObjectStorageKey = 'classificationResult';
  private readonly fileNameStorageKey = 'uploadedFileName';
  
  private probObjectSource = new BehaviorSubject<ProbObject | null>(
    this.getPropObjectFromStorage()
  );
  probObject$ = this.probObjectSource.asObservable();

  private fileNameSource = new BehaviorSubject<string | null>(
    this.getFileNameFromStorage()
  );
  fileName$ = this.fileNameSource.asObservable();

  setProbObject(probObject: ProbObject) {
    this.probObjectSource.next(probObject);
    this.savePropObjectToStorage(probObject);
  }

  getProbObject(): ProbObject | null {
    return this.probObjectSource.getValue();
  }

  setFileName(filename: string): void {
    this.fileNameSource.next(filename);
    this.saveFileNameToStorage(filename);
  }

  getFileName(): string | null {
    return this.fileNameSource.getValue();
  }

  private savePropObjectToStorage(probObject: ProbObject) {
    sessionStorage.setItem(this.propObjectStorageKey, JSON.stringify(probObject));
  }

  private saveFileNameToStorage(filename: string) {
    sessionStorage.setItem(this.fileNameStorageKey, filename);
  }

  private getPropObjectFromStorage(): ProbObject | null {
    const data = sessionStorage.getItem(this.propObjectStorageKey);
    return data ? JSON.parse(data) : null;
  }

  private getFileNameFromStorage(): string | null {
    return sessionStorage.getItem(this.fileNameStorageKey);
  }

  clearProbObject() {
    this.probObjectSource.next(null);
    sessionStorage.removeItem(this.propObjectStorageKey);
  }
  
  clearFileName() {
    this.fileNameSource.next(null);
    sessionStorage.removeItem(this.fileNameStorageKey);
  }
}
