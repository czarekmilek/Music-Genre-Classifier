import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PredictGenreComponent } from './predict-genre.component';

describe('PredictGenreComponent', () => {
  let component: PredictGenreComponent;
  let fixture: ComponentFixture<PredictGenreComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PredictGenreComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PredictGenreComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
